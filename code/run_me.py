import jax
from matplotlib import pyplot as plt
import time
import jax.numpy as jnp

# Question 1

def get_data_generator(batchsize):
    def getbatch(key):
        key, subkey = jax.random.split(key)
        context = 10*jax.random.uniform(subkey, [batchsize])
        key, subkey = jax.random.split(key)
        output = 3 + 10 * context - 0.7 * context**2 + jax.random.normal(subkey, [batchsize])
        return context, output
    return getbatch

gen = get_data_generator(100)

key = jax.random.PRNGKey(0)
context, output = gen(key)
plt.plot(context, output, 'o')
plt.xlabel('context')
plt.ylabel('output')
plt.show()

# Question 2

import time
import jax
from jax import flatten_util
from jax import numpy as jnp
from typing import Callable
def signed_gradient_descent(net: Callable, loss: Callable, getbatch: Callable, max_iter: int, learning_rates: list[int], *params):
    assert isinstance(net,Callable)
    assert isinstance(loss,Callable)
    assert isinstance(getbatch,Callable)
    assert isinstance(max_iter,int)

    key = jax.random.PRNGKey(0)
    w, unflatten = jax.flatten_util.ravel_pytree(params)

    # batch predictive network over context, but not params
    batched_net = jax.vmap(net,[0]+[None]*len(params))
    # batch loss over both context and predictions
    batched_loss = jax.vmap(loss)

    def l(key,w):
        params = unflatten(w)
        context, next = getbatch(key)
        pred = batched_net(context,*params)
        return jnp.mean(batched_loss(pred, next))

    fun = jax.value_and_grad(l,1)

    t0 = time.time()

    avg_grad = 0
    avg_loss = 0
    smooth_loss = 0

    print(" iter     l.r.     loss (smooth)    (avg)     time")
    print("----- -------- -------- -------- -------- --------")
   
    for i in range(max_iter):
        n = int(i * len(learning_rates) / max_iter)

        key, subkey = jax.random.split(key)
        loss, grad = fun(subkey, w)
        avg_loss = (i*avg_loss + loss)/(i+1)
        alpha = max(.01,1/(i+1))
        smooth_loss = alpha*loss + (1-alpha) * smooth_loss

        if i % (max_iter // 20) == 0:
            print(f"{i: >5} {learning_rates[n]:8.5f} {loss:8.5f} {smooth_loss:8.5f} {avg_loss:8.5f} {time.time()-t0:8.5f}")
        avg_grad = 0.9 * avg_grad + 0.1 * grad

        w = w - learning_rates[n]*jnp.sign(avg_grad)
   
    t = time.time()-t0
    t = round(t * 1000, 5)

    if len(params)==1:
        params = unflatten(w)[0]
        # print(f"  {t}   | {params[0]}  | {params[1]} | {params[2]} | {avg_loss} ")
        return params
    else:
        params = unflatten(w)
        # print(f"  {t}   | {params[0]}  | {params[1]} | {params[2]} | {avg_loss} ")
        return unflatten(w)

def net(context,a,b,c):
    return a + b*context + c*context**2

def loss(pred, output):
    return (pred - output)**2

# arrays of batchsize, max_iter, learning_rates
batchsizes = [1,10,100,100,100]
max_iters = 1001
learning_rates = [[.1],[.1],[.1],[.01], [.1,.01]]

print("  Time (ms)   |       a       |      b     |       c     |   final smooth loss")
print("--------------|---------------|------------|-------------|---------------------")

for batchsize, learning_rate in zip(batchsizes, learning_rates):
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, 1)
    b = jax.random.normal(key, 1)
    c = jax.random.normal(key, 1)
    params = [a,b,c]

    gen = get_data_generator(batchsize)
    key = jax.random.PRNGKey(0)
   
    params = signed_gradient_descent(net, loss, gen, max_iters, learning_rate, *params)
    print(f"  {params} ")


# Question 4

chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '!', '?', ':', '"', "'", '+', ',', '.', ' ', '\n']
start = [55, 17, 14, 70, 51, 27, 24, 19, 14, 12, 29, 70, 42, 30, 29, 14, 23,
       11, 14, 27, 16, 70, 40, 37, 24, 24, 20, 70, 24, 15, 70, 55, 17, 14,
       70, 40, 25, 18, 12, 70, 24, 15, 70, 54, 10, 30, 21, 68, 70, 11, 34,
       70, 58, 18, 21, 21, 18, 10, 22, 70, 38, 21, 14, 10, 31, 14, 27, 70,
       58, 18, 21, 20, 18, 23, 28, 24, 23]

res = ''.join([chars[c] for c in start])

print(res)

# Question 5

data = jnp.load('data.npz',mmap_mode='r')['data']

def get_data_generator(context_size, batch_size):
    def getbatch(key):
        key, subkey = jax.random.split(key)
        start = jax.random.randint(subkey, shape=batch_size, minval=0, maxval=len(data)-context_size)
        indices = start[:,None] + jnp.arange(context_size)[None,:]
        context = data[indices] # (batchsize x context_size)
        next = data[start+context_size] # (batchsize,)
        return context, next
    return getbatch

key = jax.random.PRNGKey(0)
getbatch = get_data_generator(50, 5)
context, next = getbatch(key)

print("Shape of context: ", context.shape, "Shape of next: ", next.shape)

print("context: ", context)

print("next: ", next)

# Question 6

from jax.nn import logsumexp

def loss(pred,next):
    num_characters = len(chars)
    assert pred.shape == (num_characters,)
    assert next.shape == ()
    log_probs = pred - logsumexp(pred)
    assert log_probs.shape == (num_characters,)
    next_onehot = jax.nn.one_hot(next, num_classes=num_characters)
    assert next_onehot.shape == (num_characters,)
    out = -jnp.sum(log_probs * next_onehot)
    assert out.shape == ()
    return out

def constant_net(context, b):
    # input context is a 1-D array of size context_size
    # each entry is an index between 0 and num_characters
    # these represent the most recent characters
    (context_size,) = context.shape
    (num_characters,) = b.shape

    # [do stuff.]

    # predict a constant vector
    pred = b

          
    assert pred.shape == (num_characters,)
    return pred

batchsize = 4096
context_size = 32
iters = 10000
learning_rates = [.001, .0001]
b = jnp.zeros(len(chars))

params = [b]

gen = get_data_generator(context_size, batchsize)

params = signed_gradient_descent(constant_net, loss, gen, iters, learning_rates, *params)

# Question 7

num_characters = len(chars)

char2int = dict(zip(chars, range(len(chars))))

def generate_char(net,context,key,*params):
    pred = net(context,*params);
    assert pred.shape == (num_characters,)
    out = jax.random.categorical(key, pred);
    assert out.shape == ()
    return out

def generate(net,context_str,context_size,num_char,*params):
    context = [char2int[c] for c in context_str]

    key = jax.random.PRNGKey(1)
    for i in range(num_char):
        key, subkey = jax.random.split(key)
        my_context = jnp.array(context[-context_size:])
        c = generate_char(net,my_context,subkey,*params)
        context.append(int(c))

    out = ''.join([chars[i] for i in context])

    return out

b_constant = params
start = "STUDENT:\nI have searched the skies and found...\n\nTEACHER:\nYes? What have you found?\n\nSTUDENT\nI have found a fact, a fact most excellent.\n"
print(generate(constant_net,start,context_size,500,b_constant))

# Question 8

import numpy as np

def linear_net(context, b, W):
    # W, b = params[0], params[1]
    # print('w: ', W)
    # print('b: ', b)

    (context_size,) = context.shape

    (context_size2, num_characters, num_characters2) = W.shape
    assert context_size == context_size2
    assert num_characters == num_characters2

    context_onehot = jax.nn.one_hot(context, num_classes=num_characters)
    assert context_onehot.shape == (context_size, num_characters)


    # We need to multiply context_onehot with W along the context_size dimension
    
    pred = jnp.einsum('ij,ijk->k', context_onehot, W) + b

    assert pred.shape == (num_characters,)
    return pred

batchsize = 4096
context_size = 32
iters = 10000
learning_rates = [.001, .0001]
num_characters = len(chars)

b_linear = jnp.zeros(num_characters)
W_linear = jnp.array(.01*np.random.randn(context_size, num_characters, num_characters))


params = [b_linear, W_linear]

gen = get_data_generator(context_size, batchsize)

params = signed_gradient_descent(linear_net, loss, gen, iters, learning_rates, *params)

# Question 9


p = params
start = "STUDENT:\nI have searched the skies and found...\n\nTEACHER:\nYes? What have you found?\n\nSTUDENT\nI have found a fact, a fact most excellent.\n"

print(generate(linear_net,start,context_size,500, *p))

# Question 10

def mlp_net(context, b, c, W, V):
    assert context.shape == (context_size,)
    (context_size,) = context.shape
    (num_characters,) = b.shape
    (num_hidden,) = c.shape
    (num_characters, num_hidden) = W.shape
    (context_size, num_characters, num_hidden) = V.shape

    context_onehot = jax.nn.one_hot(context, num_classes=num_characters)
    assert context_onehot.shape == (context_size, num_characters)

    # [do stuff]

    # first layer
    h1 = jnp.einsum('ij,ijk->k', context_onehot, V) + c
    l1 = jax.nn.relu(h1)
    

    pred = b + W @ l1
    assert pred.shape == (num_characters,)
    return pred

batchsize = 4096
context_size = 32
iters = 10000
learning_rates = [.001, .0001]
num_characters = len(chars)

num_hidden = 500
b_mlp = jnp.zeros(num_characters)
c_mlp = jnp.zeros(num_hidden)
W_mlp = jnp.array(.01*np.random.randn(num_characters, num_hidden))
V_mlp = jnp.array(.01*np.random.randn(context_size, num_characters, num_hidden))


params = [b_mlp, c_mlp, W_mlp, V_mlp]

gen = get_data_generator(context_size, batchsize)

params = signed_gradient_descent(mlp_net, loss, gen, iters, learning_rates, *params)

# Question 11

p = params
start = "STUDENT:\nI have searched the skies and found...\n\nTEACHER:\nYes? What have you found?\n\nSTUDENT\nI have found a fact, a fact most excellent.\n"

print(generate(mlp_net,start,context_size,500, *p))

# Question 12

def dbl_net(context, b, c, d, W, V, U):
    assert context.shape == (context_size,)

    context_onehot = jax.nn.one_hot(context, num_classes=num_characters)
    assert context_onehot.shape == (context_size, num_characters)

    # [do stuff]
    h1 = jnp.einsum('ijk,ij->k', V, context_onehot) + c
    l1 = jax.nn.relu(h1)
    
    # second layer
    h2 = jnp.einsum('k,jk->k', l1, U) + d
    l2 = jax.nn.relu(h2)

    pred = b + W @ l2
    assert pred.shape == (num_characters,)

    return pred

batchsize = 4096
context_size = 32
iters = 10000
learning_rates = [.001, .0001]
num_characters = len(chars)

num_hidden = 500
b_dbl = jnp.zeros(num_characters)
c_dbl = jnp.zeros(num_hidden)
d_dbl = jnp.zeros(num_hidden)
W_dbl = jnp.zeros((num_characters, num_hidden))
V_dbl = jnp.array(.01*np.random.randn(context_size, num_characters, num_hidden))
U_dbl = jnp.array(.01*np.random.randn(num_hidden, num_hidden))

params = [b_dbl, c_dbl, d_dbl, W_dbl, V_dbl, U_dbl]

gen = get_data_generator(context_size, batchsize)

params = signed_gradient_descent(dbl_net, loss, gen, iters, learning_rates, *params)

# Question 13

p = params
start = "STUDENT:\nI have searched the skies and found...\n\nTEACHER:\nYes? What have you found?\n\nSTUDENT\nI have found a fact, a fact most excellent.\n"

print(generate(dbl_net,start,context_size,500, *p))

# Question 14

batchsize = 4096
context_size = 32
iters = 10000
learning_rates = [.001, .0001]
num_characters = len(chars)

num_hidden = 2500
b_dbl = jnp.zeros(num_characters)
c_dbl = jnp.zeros(num_hidden)
d_dbl = jnp.zeros(num_hidden)
W_dbl = jnp.zeros((num_characters, num_hidden))
V_dbl = jnp.array(.01*np.random.randn(context_size, num_characters, num_hidden))
U_dbl = jnp.array(.01*np.random.randn(num_hidden, num_hidden))


params = [b_dbl, c_dbl, d_dbl, W_dbl, V_dbl, U_dbl]

gen = get_data_generator(context_size, batchsize)

params = signed_gradient_descent(dbl_net, loss, gen, iters, learning_rates, *params)

# Question 15

p = params
start = "STUDENT:\nI have searched the skies and found...\n\nTEACHER:\nYes? What have you found?\n\nSTUDENT\nI have found a fact, a fact most excellent.\n"

print(generate(dbl_net,start,context_size,500, *p))