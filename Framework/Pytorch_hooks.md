### 1. Forward Pre-hooks

Forward pre-hooks are executed just before the forward pass of a module. They can be attached to any `nn.Module` object using the `register_forward_pre_hook` method.

**Example**:

Let's say you want to modify the input to a particular layer:

```python
import torch.nn as nn

def forward_pre_hook(module, input):
    print(f"Inside {module.__class__.__name__}'s forward pre-hook")
    # Double the input
    return (input[0] * 2,)

model = nn.Linear(10, 20)

# Attach the pre-hook to the model
hook_handle = model.register_forward_pre_hook(forward_pre_hook)

# Run a forward pass
input_tensor = torch.rand(1, 10)
output = model(input_tensor)

# Remove the hook
hook_handle.remove()
```

### 2. Forward Hooks (Post-hooks)

These are executed immediately after the forward pass of a module. They can be used to inspect or modify the output of a module.

**Example**:

```python
def forward_hook(module, input, output):
    print(f"Inside {module.__class__.__name__}'s forward hook")
    # Add 1 to the output
    return output + 1

model = nn.Linear(10, 20)

# Attach the hook to the model
hook_handle = model.register_forward_hook(forward_hook)

# Run a forward pass
input_tensor = torch.rand(1, 10)
output = model(input_tensor)

# Remove the hook
hook_handle.remove()
```

### 3. Backward Hooks

These are executed during the backward pass. They can be used to inspect or modify gradients.

**Example**:

```python
def backward_hook(module, grad_input, grad_output):
    print(f"Inside {module.__class__.__name__}'s backward hook")
    # Zero out the gradients
    return tuple(g * 0 for g in grad_input)

model = nn.Linear(10, 20)

# Attach the hook to the model
hook_handle = model.register_backward_hook(backward_hook)

# Run a forward and backward pass
input_tensor = torch.rand(1, 10)
output = model(input_tensor)
loss = output.sum()
loss.backward()

# Remove the hook
hook_handle.remove()
```

### Use Cases for Hooks:

1. **Inspecting Activations and Gradients**: Hooks can be used to print, store, or visualize activations and gradients, aiding in debugging and understanding model behavior.
2. **Modifying Activations and Gradients**: Implement techniques like gradient clipping, gradient noise, or custom regularization.
3. **Profiling**: Measure the time taken for each layer/module during forward and backward passes.
4. **Feature Extraction**: Extract features (activations) from intermediate layers of a pre-trained model.
5. **Checking for Exploding/Vanishing Gradients**: Inspect gradients during training to diagnose training issues.

### Conclusion:

Hooks in PyTorch provide a powerful mechanism to introspect and intervene in the internal workings of models during both the forward and backward passes. They offer a high degree of flexibility, making them invaluable for debugging, custom modifications, and advanced techniques.
