import math
import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils

class LoRALinear(nn.Module):
    """LoRA wrapper for linear layers in MLX."""

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        # Initialize LoRA matrices
        # weight shape in MLX is (out_features, in_features)
        out_features, in_features = linear.weight.shape
        self.lora_A = mx.random.normal((in_features, r)) * (1.0 / math.sqrt(in_features))
        self.lora_B = mx.zeros((r, out_features))

    def __call__(self, x: mx.array) -> mx.array:
        out = self.linear(x)
        lora_out = (x @ self.lora_A) @ self.lora_B
        return out + self.scale * lora_out


def convert_to_lora(model, r: int = 8, alpha: float = 16.0):
    """
    Freeze all base parameters of the model and inject LoRA layers
    into `in_proj` and `out_proj` of each mixer block.
    """
    # 1. Freeze all parameters in the model
    for name, module in model.named_modules():
        immediate_keys = {k for k, v in module.parameters().items() if isinstance(v, mx.array)}
        module._no_grad = immediate_keys

    # 2. Inject LoRA layers
    for layer in model.layers:
        layer.mixer.in_proj = LoRALinear(layer.mixer.in_proj, r=r, alpha=alpha)
        layer.mixer.out_proj = LoRALinear(layer.mixer.out_proj, r=r, alpha=alpha)


def lora_state_dict(model):
    """Return only LoRA adapter arrays from a model."""
    flat_params = dict(utils.tree_flatten(model.parameters()))
    return {
        key: value
        for key, value in flat_params.items()
        if key.endswith(".lora_A") or key.endswith(".lora_B")
    }


def save_lora_adapters(model, path: str):
    """Save only LoRA adapter arrays to a safetensors file."""
    adapters = lora_state_dict(model)
    if not adapters:
        raise ValueError("No LoRA adapter parameters found")
    mx.save_safetensors(path, adapters)


def load_lora_adapters(model, path: str):
    """Load LoRA adapter arrays into an already LoRA-converted model."""
    if not path.endswith(".safetensors"):
        raise ValueError("Only .safetensors files are supported")

    try:
        flat_params = mx.load_safetensors(path)
    except Exception:
        import safetensors.numpy
        flat_params = safetensors.numpy.load_file(path)
        flat_params = {key: mx.array(value) for key, value in flat_params.items()}

    expected = set(lora_state_dict(model))
    incoming = set(flat_params)
    missing = expected - incoming
    extra = incoming - expected
    if missing or extra:
        raise ValueError(f"LoRA adapter key mismatch: missing={sorted(missing)}, extra={sorted(extra)}")

    model.update(utils.tree_unflatten(list(flat_params.items())))


def loss_fn(model, inputs, targets):
    """Compute the cross-entropy loss."""
    # logits shape: (B, L, V)
    logits = model(inputs)
    # targets shape: (B, L)
    loss = nn.losses.cross_entropy(logits, targets)
    return mx.mean(loss)


def make_train_step(model, optimizer):
    """Return a compiled train step function."""
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    def train_step(inputs, targets):
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        optimizer.update(model, grads)
        return loss

    # Compile the training step, explicitly tracking changing states
    return mx.compile(
        train_step,
        inputs=[model.state, optimizer.state],
        outputs=[model.state, optimizer.state]
    )
