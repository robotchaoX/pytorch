#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Resize.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <c10/util/irange.h>
#include <limits>
#include <tuple>
#include <vector>
#include <optional>
#include <algorithm>
#include <string>
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/cross_entropy_loss.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/full.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/max.h>
#include <ATen/ops/exp.h>
#include <ATen/ops/log.h>
#include <ATen/ops/logsumexp.h>
#include <ATen/ops/where.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/logical_and.h>
#include <ATen/ops/logical_or.h>
#include <ATen/ops/logical_not.h>
#include <ATen/ops/masked_fill.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/add.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/gather.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/div.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/linear_cross_entropy_backward_native.h>
#include <ATen/ops/linear_cross_entropy_native.h>
#endif

namespace at::native {

// Strategy selection for optimal chunking approach
enum class ChunkingStrategy {
    NAIVE,           // No chunking - standard approach
    VOCAB_CHUNKING,  // Chunk vocabulary dimension (existing)
    BATCH_CHUNKING   // Chunk batch dimension (new)
};

constexpr int64_t kDefaultVocabChunkSize = 4096;
constexpr int64_t kDefaultBatchChunkSize = 1024;

struct LinearCrossEntropySavedForBackward {
  Tensor logsumexp;        // [N]
  ChunkingStrategy strategy{ChunkingStrategy::NAIVE};
  int64_t chunk_size{0};
};

struct LinearCrossEntropyForwardResult {
  Tensor loss;
  std::optional<LinearCrossEntropySavedForBackward> saved;
};

Tensor batch_chunking_cpu(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size);

inline ChunkingStrategy select_chunking_strategy(c10::string_view strategy) {
  if (strategy == "none") {
    return ChunkingStrategy::NAIVE;
  } else if (strategy == "vocab") {
    return ChunkingStrategy::VOCAB_CHUNKING;
  } else if (strategy == "batch") {
    return ChunkingStrategy::BATCH_CHUNKING;
  }
  TORCH_CHECK(false,
              "Unknown chunking strategy: ",
              strategy,
              ". Valid options: 'vocab', 'batch', 'none'");
}

LinearCrossEntropyForwardResult linear_cross_entropy_forward_naive(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    bool save_for_backward);

LinearCrossEntropyForwardResult linear_cross_entropy_forward_vocab_chunking(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size,
    bool save_for_backward);

LinearCrossEntropyForwardResult linear_cross_entropy_forward_batch_chunking(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size,
    bool save_for_backward);

LinearCrossEntropyForwardResult linear_cross_entropy_forward_impl(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    ChunkingStrategy strategy,
    int64_t vocab_chunk_size,
    int64_t batch_chunk_size,
    bool save_for_backward) {

  TORCH_CHECK(input.dim() >= 2, "Expected input to have at least 2 dimensions, got ", input.dim());
  TORCH_CHECK(linear_weight.dim() == 2, "Expected linear_weight to be 2-dimensional, got ", linear_weight.dim());
  TORCH_CHECK(input.size(-1) == linear_weight.size(1),
              "Expected input.size(-1) to match linear_weight.size(1), got ",
              input.size(-1), " and ", linear_weight.size(1));

  switch (strategy) {
    case ChunkingStrategy::VOCAB_CHUNKING:
      TORCH_CHECK(
          vocab_chunk_size > 0,
          "linear_cross_entropy: vocab_chunk_size must be positive, got ",
          vocab_chunk_size);
      return linear_cross_entropy_forward_vocab_chunking(
          input,
          linear_weight,
          target,
          linear_bias_opt,
          reduction,
          ignore_index,
          label_smoothing,
          vocab_chunk_size,
          save_for_backward);
    case ChunkingStrategy::BATCH_CHUNKING:
      TORCH_CHECK(
          batch_chunk_size > 0,
          "linear_cross_entropy: batch_chunk_size must be positive, got ",
          batch_chunk_size);
      return linear_cross_entropy_forward_batch_chunking(
          input,
          linear_weight,
          target,
          linear_bias_opt,
          reduction,
          ignore_index,
          label_smoothing,
          batch_chunk_size,
          save_for_backward);
    default:
      return linear_cross_entropy_forward_naive(
          input,
          linear_weight,
          target,
          linear_bias_opt,
          reduction,
          ignore_index,
          label_smoothing,
          save_for_backward);
  }
}

Tensor linear_cross_entropy_cpu(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    c10::string_view chunking_strategy,
    std::optional<int64_t> vocab_chunk_size_opt,
    std::optional<int64_t> batch_chunk_size_opt) {

  const int64_t vocab_chunk_size = vocab_chunk_size_opt.value_or(kDefaultVocabChunkSize);
  const int64_t batch_chunk_size = batch_chunk_size_opt.value_or(kDefaultBatchChunkSize);

  auto result = linear_cross_entropy_forward_impl(
      input,
      linear_weight,
      target,
      linear_bias_opt,
      reduction,
      ignore_index,
      label_smoothing,
      select_chunking_strategy(chunking_strategy),
      vocab_chunk_size,
      batch_chunk_size,
      /*save_for_backward=*/false);

  return std::move(result.loss);
}

LinearCrossEntropyForwardResult linear_cross_entropy_forward_naive(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    bool save_for_backward) {

  auto logits = at::linear(input, linear_weight, linear_bias_opt);
  auto logits_flat = logits.reshape({-1, logits.size(-1)});
  auto target_flat = target.reshape({-1});
  auto options = logits_flat.options();
  const int64_t vocab_size = linear_weight.size(0);

  Tensor valid_mask = at::ne(target_flat, ignore_index);
  Tensor logsumexp = at::logsumexp(logits_flat, {1}, false);
  Tensor target_logits = at::zeros(logsumexp.sizes(), options);

  auto valid_indices = valid_mask.nonzero().reshape({-1});
  if (valid_indices.numel() > 0) {
    auto gathered_targets = at::index_select(target_flat, 0, valid_indices);
    auto gathered_logits = at::index_select(logits_flat, 0, valid_indices);
    auto gathered = at::gather(gathered_logits, 1, gathered_targets.unsqueeze(1)).squeeze(1);
    target_logits.scatter_(0, valid_indices, gathered);
  }

  Tensor losses = logsumexp;
  if (label_smoothing > 0.0) {
    const double smoothing = label_smoothing;
    const double uniform = smoothing / static_cast<double>(vocab_size);
    auto sum_logits = logits_flat.sum(-1);
    losses = losses - target_logits.mul(1.0 - smoothing) - sum_logits.mul(uniform);
  } else {
    losses = losses - target_logits;
  }

  losses.masked_fill_(at::logical_not(valid_mask), 0);

  Tensor loss_result;
  if (reduction == Reduction::None) {
    loss_result = losses.reshape(target.sizes());
  } else {
    auto total_loss = losses.sum();
    if (reduction == Reduction::Sum) {
      loss_result = total_loss;
    } else {
      const int64_t valid_count = valid_mask.sum().item<int64_t>();
      if (valid_count == 0) {
        loss_result = total_loss;
      } else {
        loss_result = total_loss.div(static_cast<double>(valid_count));
      }
    }
  }

  LinearCrossEntropyForwardResult result;
  result.loss = std::move(loss_result);

  if (save_for_backward) {
    LinearCrossEntropySavedForBackward saved;
    saved.logsumexp = std::move(logsumexp);
    saved.strategy = ChunkingStrategy::NAIVE;
    saved.chunk_size = logits_flat.size(0);
    result.saved = std::move(saved);
  }

  return result;
}

LinearCrossEntropyForwardResult linear_cross_entropy_forward_vocab_chunking(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size,
    bool save_for_backward) {

  auto input_flat = input.reshape({-1, input.size(-1)});
  auto target_flat = target.reshape({-1});
  const int64_t vocab_size = linear_weight.size(0);
  const auto options = input_flat.options();
  auto long_options = options.dtype(at::kLong);
  const double neg_inf = -std::numeric_limits<double>::infinity();

  Tensor running_max = at::full({input_flat.size(0)}, neg_inf, options);
  Tensor exp_sums = at::zeros({input_flat.size(0)}, options);
  Tensor target_logits = at::zeros({input_flat.size(0)}, options);
  Tensor target_found = at::zeros({input_flat.size(0)}, long_options);
  Tensor sum_logits;
  if (label_smoothing > 0.0) {
    sum_logits = at::zeros({input_flat.size(0)}, options);
  }

  Tensor valid_mask = at::ne(target_flat, ignore_index);
  const Tensor& linear_bias = linear_bias_opt.value_or(Tensor());
  const int64_t num_chunks = (vocab_size + chunk_size - 1) / chunk_size;

  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);

    auto weight_chunk = linear_weight.slice(0, start_idx, end_idx);
    std::optional<Tensor> bias_chunk;
    if (linear_bias.defined()) {
      bias_chunk = linear_bias.slice(0, start_idx, end_idx);
    }

    auto logits_chunk = at::linear(input_flat, weight_chunk, bias_chunk);

    if (label_smoothing > 0.0) {
      sum_logits = sum_logits + logits_chunk.sum(-1);
    }

    auto chunk_max = std::get<0>(logits_chunk.max(-1));
    auto new_max = at::maximum(running_max, chunk_max);

    auto exp_scale_old = at::exp(running_max.sub(new_max));
    auto shifted_logits = logits_chunk.sub(new_max.unsqueeze(-1));
    auto exp_chunk = at::sum(at::exp(shifted_logits), {-1});
    exp_sums = exp_sums.mul(exp_scale_old).add_(exp_chunk);
    running_max = new_max;

    auto lower_bound = at::ge(target_flat, start_idx);
    auto upper_bound = at::lt(target_flat, end_idx);
    auto target_chunk_mask = at::logical_and(valid_mask, lower_bound);
    target_chunk_mask = at::logical_and(target_chunk_mask, upper_bound);

    auto indices = target_chunk_mask.nonzero().reshape({-1});
    if (indices.numel() > 0) {
      auto selected_targets = at::index_select(target_flat, 0, indices);
      auto local_targets = at::sub(selected_targets, start_idx);
      auto selected_logits = at::index_select(logits_chunk, 0, indices);
    auto gathered = at::gather(selected_logits, 1, local_targets.unsqueeze(1)).squeeze(1);
    target_logits = target_logits.scatter(0, indices, gathered);
      target_found.index_put_({indices}, at::ones(indices.sizes(), long_options));
    }
  }

  auto target_found_mask = target_found.gt(0);
  auto coverage_mask = at::logical_or(target_found_mask, at::logical_not(valid_mask));
  TORCH_CHECK(
      coverage_mask.all().item<bool>(),
      "linear_cross_entropy: target index not found in vocabulary chunks");

  auto logsumexp = running_max.add(exp_sums.log());

  Tensor losses;
  if (label_smoothing > 0.0) {
    const double smoothing = label_smoothing;
    const double uniform = smoothing / static_cast<double>(vocab_size);
    auto main_term = target_logits.mul(1.0 - smoothing);
    auto uniform_term = sum_logits.mul(uniform);
    losses = logsumexp - main_term - uniform_term;
  } else {
    losses = logsumexp - target_logits;
  }

  losses.masked_fill_(at::logical_not(valid_mask), 0);

  Tensor loss_result;
  if (reduction == Reduction::None) {
    loss_result = losses.reshape(target.sizes());
  } else {
    auto total_loss = losses.sum();
    if (reduction == Reduction::Sum) {
      loss_result = total_loss;
    } else {
      const int64_t valid_count = valid_mask.sum().item<int64_t>();
      if (valid_count == 0) {
        loss_result = total_loss;
      } else {
        loss_result = total_loss.div(static_cast<double>(valid_count));
      }
    }
  }

  LinearCrossEntropyForwardResult result;
  result.loss = std::move(loss_result);

  if (save_for_backward) {
    LinearCrossEntropySavedForBackward saved;
    saved.logsumexp = std::move(logsumexp);
    saved.strategy = ChunkingStrategy::VOCAB_CHUNKING;
    saved.chunk_size = chunk_size;
    result.saved = std::move(saved);
  }

  return result;
}

LinearCrossEntropyForwardResult linear_cross_entropy_forward_batch_chunking(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size,
    bool save_for_backward) {

  auto input_flat = input.reshape({-1, input.size(-1)});
  auto target_flat = target.reshape({-1});
  const int64_t batch_size = input_flat.size(0);
  const Tensor& linear_bias = linear_bias_opt.value_or(Tensor());

  Tensor valid_mask = at::ne(target_flat, ignore_index);
  Tensor logsumexp = at::empty({batch_size}, input.options());
  Tensor target_logits = at::zeros({batch_size}, input.options());
  Tensor sum_logits;
  if (label_smoothing > 0.0) {
    sum_logits = at::empty({batch_size}, input.options());
  }

  Tensor losses_buffer;
  if (reduction == Reduction::None) {
    losses_buffer = at::empty({batch_size}, input.options());
  }

  Tensor total_loss = at::zeros({}, input.options());
  int64_t valid_count = 0;

  TORCH_CHECK(
      chunk_size > 0,
      "linear_cross_entropy: batch_chunk_size must be positive, got ",
      chunk_size);

  const int64_t num_chunks = (batch_size + chunk_size - 1) / chunk_size;

  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t slice = std::min<int64_t>(chunk_size, batch_size - start_idx);
    if (slice <= 0) {
      continue;
    }

    auto input_chunk = input_flat.narrow(0, start_idx, slice);
    auto target_chunk = target_flat.narrow(0, start_idx, slice);
    auto valid_mask_chunk = valid_mask.narrow(0, start_idx, slice);

    auto logits_chunk = at::linear(input_chunk, linear_weight, linear_bias);
    auto logsumexp_chunk = at::logsumexp(logits_chunk, {1}, false);
    logsumexp.narrow(0, start_idx, slice).copy_(logsumexp_chunk);

    auto valid_indices = valid_mask_chunk.nonzero().reshape({-1});
    if (valid_indices.numel() > 0) {
      auto local_targets = at::index_select(target_chunk, 0, valid_indices);
      auto selected_logits = at::index_select(logits_chunk, 0, valid_indices);
      auto gathered = at::gather(selected_logits, 1, local_targets.unsqueeze(1)).squeeze(1);
      auto shifted_indices = valid_indices.add(start_idx);
      target_logits.scatter_(0, shifted_indices, gathered);
    }

    if (label_smoothing > 0.0) {
      auto chunk_sum = logits_chunk.sum(-1);
      sum_logits.narrow(0, start_idx, slice).copy_(chunk_sum);
    }

    const auto ce_reduction = (reduction == Reduction::None) ? Reduction::None : Reduction::Sum;
    auto chunk_loss = at::cross_entropy_loss(
        logits_chunk,
        target_chunk,
        std::nullopt,
        ce_reduction,
        ignore_index,
        label_smoothing);

    valid_count += valid_mask_chunk.sum().item<int64_t>();
    if (reduction == Reduction::None) {
      auto dest = losses_buffer.narrow(0, start_idx, slice);
      dest.copy_(chunk_loss);
      dest.masked_fill_(at::logical_not(valid_mask_chunk), 0);
    } else {
      total_loss = total_loss.add(chunk_loss);
    }
  }

  Tensor loss_result;
  if (reduction == Reduction::None) {
    loss_result = losses_buffer.reshape(target.sizes());
  } else if (reduction == Reduction::Sum || valid_count == 0) {
    loss_result = total_loss;
  } else {
    loss_result = total_loss.div(static_cast<double>(valid_count));
  }

  LinearCrossEntropyForwardResult result;
  result.loss = std::move(loss_result);

  if (save_for_backward) {
    LinearCrossEntropySavedForBackward saved;
    saved.logsumexp = std::move(logsumexp);
    saved.strategy = ChunkingStrategy::BATCH_CHUNKING;
    saved.chunk_size = chunk_size;
    result.saved = std::move(saved);
  }

  return result;
}

// The backward implementation mirrors the forward chunking strategy so we never
// materialize a full [N, vocab] tensor while reconstructing gradients.  The
// helpers below break the work into vocabulary-chunked and batch-chunked paths
// and rely exclusively on ATen operators so that the code remains device agnostic
// and benefits from existing BLAS/cuBLAS bindings.
namespace {

inline Tensor zeros_like_tensor(const Tensor& src) {
  return at::_ops::zeros_like::call(src, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
}

inline Tensor zeros_like_or_undef(const std::optional<Tensor>& opt) {
  if (opt.has_value()) {
    return zeros_like_tensor(opt.value());
  }
  return Tensor();
}

inline Tensor cast_grad_output(const Tensor& grad_output, const Tensor& input) {
  return grad_output.to(input.scalar_type());
}

inline Tensor mask_invalid_rows(const Tensor& tensor, const Tensor& valid_mask) {
  auto mask = valid_mask.to(tensor.scalar_type()).unsqueeze(1);
  return at::mul(tensor, mask);
}

inline void apply_target_updates(
    Tensor& grad_chunk,
    const Tensor& target_flat,
    const Tensor& rows,
    int64_t offset,
    double label_smoothing) {
  if (rows.numel() == 0) {
    return;
  }
  auto selected_targets = at::index_select(target_flat, 0, rows);
  auto local_targets = selected_targets.add(-offset).to(at::kLong);
  auto gather = grad_chunk.index({rows, local_targets}).add(-(1.0 - label_smoothing));
  grad_chunk.index_put_({rows, local_targets}, gather);
}

inline void scale_grad_chunk(
    Tensor& grad_chunk,
    const Tensor& grad_output_tensor,
    const Tensor& grad_output_flat,
    int64_t reduction,
    int64_t valid_count) {
  if (reduction == Reduction::None) {
    grad_chunk.mul_(grad_output_flat.unsqueeze(1));
    return;
  }
  if (reduction == Reduction::Sum) {
    grad_chunk.mul_(grad_output_tensor);
    return;
  }
  TORCH_CHECK(valid_count >= 0, "Valid element count must be non-negative");
  if (valid_count == 0) {
    grad_chunk.zero_();
    return;
  }
  auto scale = grad_output_tensor.div(static_cast<double>(valid_count));
  grad_chunk.mul_(scale);
}

// Computes gradients when the forward pass chose vocabulary chunking.  We run a
// first pass to rebuild the per-sample logsumexp using the same streaming scheme
// as the forward kernel, then revisit each chunk to accumulate gradients for the
// input, weight and (optional) bias tensors.
inline std::tuple<Tensor, Tensor, std::optional<Tensor>> backward_vocabulary_chunking(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    const Tensor& grad_output,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size,
    const Tensor& saved_logsumexp,
    bool has_saved) {
  const auto input_flat = input.reshape({-1, input.size(-1)});
  const auto target_flat = target.reshape({-1});
  const auto dtype = input.scalar_type();
  const auto options = input.options();
  Tensor valid_mask = at::ne(target_flat, ignore_index);
  const int64_t valid_count = valid_mask.sum().item<int64_t>();

  if (reduction == Reduction::Mean && valid_count == 0) {
    Tensor grad_input = zeros_like_tensor(input);
    Tensor grad_weight = zeros_like_tensor(linear_weight);
    Tensor grad_bias = zeros_like_or_undef(linear_bias_opt);
    std::optional<Tensor> grad_bias_opt;
    if (grad_bias.defined()) {
      grad_bias_opt = std::move(grad_bias);
    }
    return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias_opt));
  }

  TORCH_CHECK(chunk_size > 0, "linear_cross_entropy: vocab_chunk_size must be positive, got ", chunk_size);
  const int64_t vocab_size = linear_weight.size(0);
  const int64_t num_chunks = (vocab_size + chunk_size - 1) / chunk_size;

  Tensor logsumexp;
  if (has_saved) {
    logsumexp = saved_logsumexp;
  } else {
    Tensor running_max = at::full({input_flat.size(0)}, -std::numeric_limits<double>::infinity(), options).to(dtype);
    Tensor exp_sums = at::zeros({input_flat.size(0)}, options).to(dtype);

    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      const int64_t start_idx = chunk_idx * chunk_size;
      const int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);
      auto weight_chunk = linear_weight.slice(0, start_idx, end_idx);
      std::optional<Tensor> bias_chunk;
      if (linear_bias_opt.has_value()) {
        bias_chunk = linear_bias_opt->slice(0, start_idx, end_idx);
      }
      auto logits_chunk = at::linear(input_flat, weight_chunk, bias_chunk);
      auto chunk_max = std::get<0>(logits_chunk.max(-1));
      auto new_max = at::maximum(running_max, chunk_max);
      auto exp_scale_old = at::exp(running_max.sub(new_max));
      auto shifted_logits = logits_chunk.sub(new_max.unsqueeze(-1));
      auto exp_chunk = at::sum(at::exp(shifted_logits), {-1});
      exp_sums = exp_sums.mul(exp_scale_old).add_(exp_chunk);
      running_max = new_max;
    }

    logsumexp = running_max.add(exp_sums.log());
  }

  Tensor grad_input = zeros_like_tensor(input_flat);
  Tensor grad_weight = zeros_like_tensor(linear_weight);
  Tensor grad_bias = zeros_like_or_undef(linear_bias_opt);

  const double uniform_component = label_smoothing > 0.0 ? label_smoothing / static_cast<double>(vocab_size) : 0.0;
  Tensor grad_output_tensor = cast_grad_output(grad_output, input);
  Tensor grad_output_flat;
  if (reduction == Reduction::None) {
    grad_output_flat = grad_output_tensor.reshape(-1);
  }

  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t end_idx = std::min(start_idx + chunk_size, vocab_size);
    auto weight_chunk = linear_weight.slice(0, start_idx, end_idx);
    std::optional<Tensor> bias_chunk;
    if (linear_bias_opt.has_value()) {
      bias_chunk = linear_bias_opt->slice(0, start_idx, end_idx);
    }
    auto logits_chunk = at::linear(input_flat, weight_chunk, bias_chunk);
    auto grad_chunk = at::exp(logits_chunk.sub(logsumexp.unsqueeze(-1)));
    if (label_smoothing > 0.0) {
      grad_chunk = grad_chunk.add(-uniform_component);
    }
    grad_chunk = mask_invalid_rows(grad_chunk, valid_mask);
    auto lower_bound = target_flat.ge(start_idx);
    auto upper_bound = target_flat.lt(end_idx);
    auto target_chunk_mask = at::logical_and(valid_mask, lower_bound);
    target_chunk_mask = at::logical_and(target_chunk_mask, upper_bound);
    auto rows = target_chunk_mask.nonzero().squeeze(-1);
    apply_target_updates(grad_chunk, target_flat, rows, start_idx, label_smoothing);
    scale_grad_chunk(grad_chunk, grad_output_tensor, grad_output_flat, reduction, valid_count);
    grad_chunk = mask_invalid_rows(grad_chunk, valid_mask);
    grad_input.add_(grad_chunk.matmul(weight_chunk));
    grad_weight.slice(0, start_idx, end_idx).add_(grad_chunk.transpose(0, 1).matmul(input_flat));
    if (grad_bias.defined()) {
      grad_bias.slice(0, start_idx, end_idx).add_(grad_chunk.sum(0));
    }
  }

  grad_input = grad_input.reshape_as(input);
  std::optional<Tensor> grad_bias_opt;
  if (grad_bias.defined()) {
    grad_bias_opt = std::move(grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias_opt));
}

inline std::tuple<Tensor, Tensor, std::optional<Tensor>> backward_batch_chunking(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    const Tensor& grad_output,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    int64_t chunk_size,
    const Tensor& saved_logsumexp,
    bool has_saved) {
  const auto input_flat = input.reshape({-1, input.size(-1)});
  const auto target_flat = target.reshape({-1});
  Tensor valid_mask = at::ne(target_flat, ignore_index);
  const int64_t valid_count = valid_mask.sum().item<int64_t>();

  if (reduction == Reduction::Mean && valid_count == 0) {
    Tensor grad_input = zeros_like_tensor(input);
    Tensor grad_weight = zeros_like_tensor(linear_weight);
    Tensor grad_bias = zeros_like_or_undef(linear_bias_opt);
    std::optional<Tensor> grad_bias_opt;
    if (grad_bias.defined()) {
      grad_bias_opt = std::move(grad_bias);
    }
    return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias_opt));
  }

  const auto options = input.options();
  const Tensor& linear_bias = linear_bias_opt.value_or(Tensor());

  TORCH_CHECK(chunk_size > 0, "linear_cross_entropy: batch_chunk_size must be positive, got ", chunk_size);
  const int64_t total = input_flat.size(0);
  const int64_t num_chunks = (total + chunk_size - 1) / chunk_size;

  Tensor logsumexp;
  if (has_saved) {
    logsumexp = saved_logsumexp;
  } else {
    logsumexp = at::empty({total}, options);

    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      const int64_t start_idx = chunk_idx * chunk_size;
      const int64_t slice = std::min<int64_t>(chunk_size, total - start_idx);
      if (slice <= 0) {
        continue;
      }
      auto input_chunk = input_flat.narrow(0, start_idx, slice);
      auto logits_chunk = at::linear(input_chunk, linear_weight, linear_bias);
      auto logsumexp_chunk = at::logsumexp(logits_chunk, {1}, false);
      logsumexp.narrow(0, start_idx, slice).copy_(logsumexp_chunk);
    }
  }

  Tensor grad_input = zeros_like_tensor(input_flat);
  Tensor grad_weight = zeros_like_tensor(linear_weight);
  Tensor grad_bias = zeros_like_or_undef(linear_bias_opt);

  Tensor grad_output_tensor = cast_grad_output(grad_output, input);
  Tensor grad_output_flat;
  if (reduction == Reduction::None) {
    grad_output_flat = grad_output_tensor.reshape(-1);
  }

  const double uniform_component = label_smoothing > 0.0
      ? label_smoothing / static_cast<double>(linear_weight.size(0))
      : 0.0;

  for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const int64_t start_idx = chunk_idx * chunk_size;
    const int64_t slice = std::min<int64_t>(chunk_size, total - start_idx);
    if (slice <= 0) {
      continue;
    }
    auto input_chunk = input_flat.narrow(0, start_idx, slice);
    auto target_chunk = target_flat.narrow(0, start_idx, slice);
    auto valid_mask_chunk = valid_mask.narrow(0, start_idx, slice);
    auto logits_chunk = at::linear(input_chunk, linear_weight, linear_bias_opt);
    auto logsumexp_chunk = logsumexp.narrow(0, start_idx, slice);
    auto grad_chunk = at::exp(logits_chunk.sub(logsumexp_chunk.unsqueeze(-1)));
    if (label_smoothing > 0.0) {
      grad_chunk = grad_chunk.add(-uniform_component);
    }
    grad_chunk = mask_invalid_rows(grad_chunk, valid_mask_chunk);
    auto rows = valid_mask_chunk.nonzero().squeeze(-1);
    if (rows.numel() > 0) {
      auto targets_slice = at::index_select(target_chunk, 0, rows).to(at::kLong);
      auto gather = grad_chunk.index({rows, targets_slice}).add(-(1.0 - label_smoothing));
      grad_chunk.index_put_({rows, targets_slice}, gather);
    }
    if (reduction == Reduction::None) {
      grad_chunk.mul_(grad_output_flat.narrow(0, start_idx, slice).unsqueeze(1));
    } else if (reduction == Reduction::Sum) {
      grad_chunk.mul_(grad_output_tensor);
    } else {
      if (valid_count == 0) {
        continue;
      }
      grad_chunk.mul_(grad_output_tensor.div(static_cast<double>(valid_count)));
    }
    grad_chunk = mask_invalid_rows(grad_chunk, valid_mask_chunk);
    grad_input.narrow(0, start_idx, slice).add_(grad_chunk.matmul(linear_weight));
    grad_weight.add_(grad_chunk.transpose(0, 1).matmul(input_chunk));
    if (grad_bias.defined()) {
      grad_bias.add_(grad_chunk.sum(0));
    }
  }

  grad_input = grad_input.reshape_as(input);
  std::optional<Tensor> grad_bias_opt;
  if (grad_bias.defined()) {
    grad_bias_opt = std::move(grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias_opt));
}

} // anonymous namespace

std::tuple<Tensor, Tensor, std::optional<Tensor>> linear_cross_entropy_backward_cpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    c10::string_view chunking_strategy,
    std::optional<int64_t> vocab_chunk_size_opt,
    std::optional<int64_t> batch_chunk_size_opt) {

  TORCH_CHECK(input.dim() >= 2, "Expected input to have at least 2 dimensions, got ", input.dim());
  TORCH_CHECK(linear_weight.dim() == 2, "Expected linear_weight to be 2-dimensional, got ", linear_weight.dim());
  TORCH_CHECK(input.size(-1) == linear_weight.size(1),
      "Expected input.size(-1) to match linear_weight.size(1), got ",
      input.size(-1), " and ", linear_weight.size(1));
  TORCH_CHECK(target.device() == input.device(), "Target must be on the same device as input");

  const int64_t flattened_batch = input.numel() / input.size(-1);
  ChunkingStrategy resolved_strategy = select_chunking_strategy(chunking_strategy);

  if (resolved_strategy == ChunkingStrategy::VOCAB_CHUNKING) {
    return backward_vocabulary_chunking(
        input,
        linear_weight,
        target,
        linear_bias_opt,
        grad_output,
        reduction,
        ignore_index,
        label_smoothing,
        vocab_chunk_size_opt.value_or(kDefaultVocabChunkSize),
        Tensor(),
        /*has_saved=*/false);
  }

  const int64_t default_chunk = resolved_strategy == ChunkingStrategy::BATCH_CHUNKING
      ? batch_chunk_size_opt.value_or(kDefaultBatchChunkSize)
      : std::max<int64_t>(flattened_batch, 1);
  return backward_batch_chunking(
      input,
      linear_weight,
      target,
      linear_bias_opt,
      grad_output,
      reduction,
      ignore_index,
      label_smoothing,
      default_chunk,
      Tensor(),
      /*has_saved=*/false);
}

namespace {

class LinearCrossEntropyAutogradFunction
    : public torch::autograd::Function<LinearCrossEntropyAutogradFunction> {
 public:
  static Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& input,
      const Tensor& linear_weight,
      const Tensor& target,
      const Tensor& linear_bias,
      int64_t reduction,
      int64_t ignore_index,
      double label_smoothing,
      int64_t strategy_code,
      int64_t vocab_chunk_size,
      int64_t batch_chunk_size) {

    std::optional<Tensor> linear_bias_opt;
    if (linear_bias.defined()) {
      linear_bias_opt = linear_bias;
    }

    ChunkingStrategy strategy = static_cast<ChunkingStrategy>(strategy_code);
    auto forward_result = linear_cross_entropy_forward_impl(
        input,
        linear_weight,
        target,
        linear_bias_opt,
        reduction,
        ignore_index,
        label_smoothing,
        strategy,
        vocab_chunk_size,
        batch_chunk_size,
        /*save_for_backward=*/true);

    TORCH_CHECK(forward_result.saved.has_value(), "linear_cross_entropy: expected saved data for backward");
    const auto& saved = forward_result.saved.value();

    auto long_opts = input.options().device(at::kCPU).dtype(at::kLong);
    auto double_opts = input.options().device(at::kCPU).dtype(at::kDouble);

    std::vector<Tensor> tensors_to_save;
    tensors_to_save.reserve(12);
    tensors_to_save.push_back(input);
    tensors_to_save.push_back(linear_weight);
    tensors_to_save.push_back(target);
    tensors_to_save.push_back(linear_bias);
    tensors_to_save.push_back(saved.logsumexp);
    tensors_to_save.push_back(at::full({}, static_cast<int64_t>(saved.strategy), long_opts));
    tensors_to_save.push_back(at::full({}, saved.chunk_size, long_opts));
    tensors_to_save.push_back(at::full({}, reduction, long_opts));
    tensors_to_save.push_back(at::full({}, ignore_index, long_opts));
    tensors_to_save.push_back(at::full({}, vocab_chunk_size, long_opts));
    tensors_to_save.push_back(at::full({}, batch_chunk_size, long_opts));
    tensors_to_save.push_back(at::full({}, label_smoothing, double_opts));

    ctx->save_for_backward(tensors_to_save);
    return forward_result.loss;
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto it = saved.begin();
    Tensor input = *it++;
    Tensor linear_weight = *it++;
    Tensor target = *it++;
    Tensor linear_bias_saved = *it++;
    Tensor logsumexp = *it++;
    Tensor strategy_tensor = *it++;
    Tensor chunk_tensor = *it++;
    Tensor reduction_tensor = *it++;
    Tensor ignore_tensor = *it++;
    Tensor vocab_chunk_tensor = *it++;
    Tensor batch_chunk_tensor = *it++;
    Tensor smoothing_tensor = *it++;

    std::optional<Tensor> linear_bias_opt;
    if (linear_bias_saved.defined()) {
      linear_bias_opt = linear_bias_saved;
    }

    const int64_t strategy_code = strategy_tensor.item<int64_t>();
    const int64_t chunk_size_saved = chunk_tensor.item<int64_t>();
    const int64_t reduction = reduction_tensor.item<int64_t>();
    const int64_t ignore_index = ignore_tensor.item<int64_t>();
    const int64_t vocab_chunk_size = vocab_chunk_tensor.item<int64_t>();
    const int64_t batch_chunk_size = batch_chunk_tensor.item<int64_t>();
    const double label_smoothing = smoothing_tensor.item<double>();

    Tensor grad_output = grad_outputs[0];

    const int64_t flattened_batch = input.numel() / input.size(-1);

    Tensor grad_input;
    Tensor grad_weight;
    std::optional<Tensor> grad_bias_opt;

    ChunkingStrategy strategy = static_cast<ChunkingStrategy>(strategy_code);
    const int64_t effective_vocab_chunk = chunk_size_saved > 0 ? chunk_size_saved : vocab_chunk_size;
    const int64_t effective_batch_chunk = chunk_size_saved > 0 ? chunk_size_saved : batch_chunk_size;

    if (strategy == ChunkingStrategy::VOCAB_CHUNKING) {
      auto grads = backward_vocabulary_chunking(
          input,
          linear_weight,
          target,
          linear_bias_opt,
          grad_output,
          reduction,
          ignore_index,
          label_smoothing,
          effective_vocab_chunk,
          logsumexp,
          /*has_saved=*/true);
      grad_input = std::get<0>(grads);
      grad_weight = std::get<1>(grads);
      grad_bias_opt = std::get<2>(grads);
    } else {
      auto grads = backward_batch_chunking(
          input,
          linear_weight,
          target,
          linear_bias_opt,
          grad_output,
          reduction,
          ignore_index,
          label_smoothing,
          strategy == ChunkingStrategy::BATCH_CHUNKING ? effective_batch_chunk : flattened_batch,
          logsumexp,
          /*has_saved=*/true);
      grad_input = std::get<0>(grads);
      grad_weight = std::get<1>(grads);
      grad_bias_opt = std::get<2>(grads);
    }

    torch::autograd::tensor_list outputs(10);
    outputs[0] = grad_input;
    outputs[1] = grad_weight;
    outputs[2] = Tensor();
    outputs[3] = grad_bias_opt.has_value() ? grad_bias_opt.value() : Tensor();
    outputs[4] = Tensor();
    outputs[5] = Tensor();
    outputs[6] = Tensor();
    outputs[7] = Tensor();
    outputs[8] = Tensor();
    outputs[9] = Tensor();
    return outputs;
  }
};

} // anonymous namespace

Tensor linear_cross_entropy_autograd_dispatch(
    const Tensor& input,
    const Tensor& linear_weight,
    const Tensor& target,
    const std::optional<Tensor>& linear_bias_opt,
    int64_t reduction,
    int64_t ignore_index,
    double label_smoothing,
    c10::string_view chunking_strategy,
    std::optional<int64_t> vocab_chunk_size_opt,
    std::optional<int64_t> batch_chunk_size_opt) {

  Tensor bias = linear_bias_opt.has_value() ? linear_bias_opt.value() : Tensor();
  const int64_t vocab_chunk_size = vocab_chunk_size_opt.value_or(kDefaultVocabChunkSize);
  const int64_t batch_chunk_size = batch_chunk_size_opt.value_or(kDefaultBatchChunkSize);
  const int64_t strategy_code = static_cast<int64_t>(select_chunking_strategy(chunking_strategy));

  return LinearCrossEntropyAutogradFunction::apply(
      input,
      linear_weight,
      target,
      bias,
      reduction,
      ignore_index,
      label_smoothing,
      strategy_code,
      vocab_chunk_size,
      batch_chunk_size);
}

TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  m.impl("linear_cross_entropy", TORCH_FN(linear_cross_entropy_autograd_dispatch));
}

} // namespace at::native
