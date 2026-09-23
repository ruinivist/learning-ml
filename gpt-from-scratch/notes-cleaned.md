# GPT from scratch notes

## Foundations

### Context size

what max tokens ( or chars if the model is char based ) are on the input side

in char model, if context size is 8 => up to 8 chars can be used to predict the 9th

### Embedding

an embedding is just a map from token ( here char int ids ) to a vector rep of it, just raw matrices, nothing extra

### The split

data is a matrix, one int for each char

randomise?
the text will be mangled

chunk and randomise?
at boundaries there will be leakage as training set
ends up in val, since val is much smaller this is a problem

why randomise in general?
distribution bias in data, not much relevant here

### Bigram model

predicts next char from ONLY the current char, ignores rest of context
table is (vocab_size, vocab_size): row = current char, col = score for next char
emb in general is (vocab size, embedding dims), for a bigram the
2nd dim equals the first

classical version: count[a][b] / sum(count[a]) -> literally the frequency,
this IS the exact max-likelihood solution, closed form
count is (vocab,vocab) -> for each a -> b you inc count[a][b] by 1
these will be raw nums, then to get probabs, you divided it by total
where a is the first char

neural version: nn.Embedding(65, 65), random init, trained w/ gradient descent
+ cross entropy. same optimization problem, converges to the SAME answer as
counting (loss surface here has no hidden layers/nonlinearity, so it's convex,
like logistic regression per row) -> counting isn't "truth" vs GD "approx",
they're two ways to solve for the same optimum

## Information theory

### Entropy, cross entropy, and KL divergence

need to cover some topics like entropy, cross entropy, kl divergence

How many bits do u need to encode a dice throw, obviously $\log_2(2)$ = 1
what about a 3 sided unbiased dice? theoretical optimal is $\log_2(3)$ ~= 1.59
for just one outcome, you can do no better than ceil = 4

But the idea here is not of one roll, you can take as many rolls as possible
group them into one and then find the average per roll
say I take 3
$3^3$ => 27 outcomes => $\log_2(27)$ ~= 4.75 bits
say I take a ceil here to actually be able to encode it => 5 bits
5 bits per 3 rolls => 1.67 bits per roll

carrying on till inf will get you closer to the theoretical limit
this is the idea of that fractional entropy bits, you can do NO better than
that

note that thus $\log_2(\text{counts})$ is just the general case? what if probs are
unequal?

$$H(p) = \text{entropy (Shannon entropy)} = -\sum_x p(x)\log_2 p(x)$$

Now this negative and all looks a bit weird but note that probabs are all less
than one so inner terms are negative, what $H(p)$ will give you is the average
number of bits => proven to be the theoretical minimum ( the proof I'll skip )
for encoding some outcomes, biased or unbiased

for the 3 die case, each p(x) being 1/3
this is $3 \times -\frac{1}{3}\log_2\!\left(\frac{1}{3}\right) \Rightarrow -\log_2(3^{-1}) \Rightarrow \log_2(3)$

Relating to $I(x)$
$I(x) = -\log p(x)$ => how rare is $p$, the smaller it is, the more bits we need to
encode it, so in a way it encodes surprise -> information it carries.
for a certain event this is 0

so $H(p) = \sum_x p(x) \cdot I(x)$

say the task is under limited constraints assign bit count to $I(x)$
$H(p)$ is the theoretical minima, if you make a mistake and use a wrong $I'$ vector
you get to $H(p,q) = \sum_x p(x) \cdot I_q(x)$
this is higher than minima, this $H(p,q)$ is the cross-entropy
where you GUESS using $q$, and then try to compare how bad you made it by seeing
known true values

the cross entropy "loss" is case where $p$ is `y_true` and $q$ is `y_preds`

KL divergence => $\mathrm{KL}(p\|q)$ is simply the extra cost from the wrong belief

$$H(p,q) - H(p) \Rightarrow \sum_x p(x)\log\!\left(\frac{p(x)}{q(x)}\right)$$

so cross entropy is -> total cost paid
entropy -> cost if you are right
KL divergence -> additional cost paid

same as cross entropy, KL div is also not symmetric in p and q

I don't have an intuition yet but let's stop here for this is somewhat off
track already.

### Perplexity

apparently it's a real thing

take entropy as $H(p)$, you get number of bits but the dist may not be uniform
ask what number of outcomes for an equivalent uniform distribution do I need
for it to have the same entropy $\log_2(N)$ so this is just $\exp(H(p))$
this number is perplexity

so perplexity of 11 = same uncertainty as picking from a bag of 11 chars

## PyTorch mechanics

### PyTorch addressing modes

commas separate DIMENSIONS, not elements. `t[1, 2]` = row 1, col 2.
on a 1-d tensor `t[3,]` is the tuple `(3,)` => same as `t[3]`, not a slice.

on t of shape `(4, 5)`:

| form         | example   | shape     | note                         |
|--------------|-----------|-----------|------------------------------|
| int          | t[1]      | (5,)      | DROPS the dim                |
| slice        | t[1:3]    | (2, 5)    | KEEPS the dim                |
| slice to end | t[1:]     | (3, 5)    | colon, not comma             |
| per-dim      | t[1, 2]   | ()        | scalar tensor                |
| row          | t[1, :]   | (5,)      |                              |
| column       | t[:, 2]   | (4,)      |                              |
| ellipsis     | t[..., 2] | (4,)      | last dim, whatever precedes  |
| newaxis      | t[None]   | (1, 4, 5) | INSERTS a dim                |
|              | t[:, None]| (4, 1, 5) | insert anywhere              |
| bool mask    | t[t > 0]  | (k,)      | flattens to the matches      |
| index tensor | t[[0, 3]] | (2, 5)    | gathers those rows, in order |

views vs copies:
basic indexing (int, slice, ellipsis, None) -> VIEW, shares memory,
writing to it mutates the original
advanced indexing (bool mask, index tensor) -> COPY

index tensors are the embedding lookup:
`emb_table` `(vocab_size, n_embd)` indexed by ids `(B, T)` -> `(B, T, n_embd)`
the index tensor's shape REPLACES the dim it indexes. that's `nn.Embedding`.

### nn.Embedding

(key count, dim of each embedding)

B, T, C => batch, time, channels
this convention is now a covered mix of different types of models

my input => (B, T) = (32, 8)
B batch, T is position in sequence ( time ), for example of the 8 chars the
ith char
C is channels => what is the embedding for EACH such char ( cell ) identified
by (B,T) in input, this (B,T) is a single char, what passing through
token embedding does it expand each cell to C channels, so that 'c' ( int )
here becomes embedding of c -> 65 dim vec ( 65 as in bigram model the embed
dim is same as the vocab size )
so B, T, C => (32, 8, 65)

### View

(tensor).view( new shape <- can be a tuple or , separated dims )
this view function is available on any tensor

a single -1 is allowed => make it same as above dim, but calc the -1 dim from
surrounding elems so match totals

```text
say tensor is (a,b,c,d)
(a,-1,d) => (a,bc,d)
(a,-1) => (a,bcd)
(a,b,-1) => (a,b,cd)
(-1) => (abcd)
(-1,d) => (abc, d)
```

### Logits

historically logit was from stats, as the logit func

$$\operatorname{logit}(p) = \log\!\left(\frac{p}{1-p}\right)$$

it's inverse of sigmoid ( real to probab range ) so does
(0,1) to real

now it's not that same logic func but it is just used for an unbounded range
that WILL later on be converted to probabs using softmax etc

nats => $\ln$ => where we use $\ln$ instead of $\log_2$ for information bits
nats => natural units ( of information )
nothing changes just you use a diff unit so all results change by a constant
technically as we are using a different "unit"
Now each "bit" has "e" units of information

Assume $p(y\mid x)$ => char y after x is same for all => 1/65
we would reach to logits as ln(number of choices) = $\ln(65)$ ~= 4.17

this 4.17 is $H(p,q) \to H(p)$

Gibbs' inequality $H(p,q) \ge H(p,p)$
ALWAYS and it's more related to mathematics
so in here my torch embedding chose from $N(0,1)$ and seems to be in ballpark
of 4.4 to 4.8 which makes sense, it's > 4.17

Convex function -> draw a chord along any two points on the curve, if it's
ABOVE curve always it's convex
this is the better check as I always had an incorrect notion of "sloping up"

Jensen's inequality
$f(\text{averages}) \le \text{average}(f)$
for any set of x points on the curve provided f is convex
naturally, this flips if f is concave

### The three attributes for nn.Module

the `__setattr__` is overridden so an assignment if it's a
`nn.Parameter` or another module it gets parameters + auto move to devices +
state dict etc
if you just assign random tensor, nothing of that sort
but say it's a buffer, some tensor you want to not have grads but register as
part of the model ( so that move, state dicts all work but no grads ) this is
what `register_buffer` is for

### More of torch

a dot is idiomatically `a @ b.transpose(dim a index, dim b index)` where the
result will have these dimensions swapped
why not `torch.dot` -> would do it on 1d vectors, literally dot, for across
batch you would need to call it batch times, doing it as single matmuls which
are heavily auto optimised is always better and general rule in torch

`transpose( a dim index, b dim index )`
swaps out these two dimensions, since swap is symmetric `transpose(a,b) = transpose(b,a)`

we earlier did `tril == 0` for the mask step making it `triu(1)` 1 meaning
diagonals are zeroed too

### Broadcasting rules

same in pytorch as they are in numpy
*todo*

### Training

the iteration is always
```python
logits, loss = model(x,y)
opt.zero_grad()
loss.backward()
opt.step()
```

the gradients live in each parameter
```python
model.token_embedding.weight.grad
```

rem, backward on loss and rest on opts
you never call forward directly

why does loss.backward not do zero grad itself?
The idea was loss accumulation, say you can only run forward for batch size 4
at a time but want the step to be equivalent of 32 batch size, then you don't
clear for 8 such runs, making the loss gradients accumulate before you step

Note that here the loss is on just one batch, not global and can even be less
than the global optimal floor over full dataset

### Estimate loss

once training is done, you estimate loss over train as well as eval, averaging
over some small number of iters

wrap in `@torch.no_grad()` decorator so that forward does not build any
autograd graph
also `model.eval()` which changes behaviour of some layers like dropout but we
don't use any here so doesn't matter

### Generation and sampling

`[]` indexing of tensors
in this you give a comma separated list of SELECTORS ( not real term )
selectors are assigned left to right, a missing one is just ":"
for each selector it can be
":" -> all
"2:4", index 2 inc to 3 excl
"-1" -> just the last one, negative indices work obviously

> for slicing, "x:" is same as "x:None"

logits is (B, T, C), only care about newest prediction = last time step

```text
logits[:, -1, :]
  :  on dim0 -> keep whole batch dim, untouched
 -1  on dim1 -> INT index, so it DROPS this dim, keeps only last time step
  :  on dim2 -> keep all C channels, untouched
 (B, T, C) -> (B, C)  # dim1 gone entirely, not size 1, GONE as you select
 a specific index, if you need it need to select as slice of len 1
```

### Softmax

converting logits to preds
say v is the vector of logits

$$p(i) = \frac{\exp(v_i)}{\sum_j \exp(v_j)}$$

### Multinomial selection

weight by preds and select one sample

### Cat

append that to context at 1 so the context var keeps on growing but ofc
we don't use the earlier vals, or in fact any but the last

## Attention

Starting with an example

say we are at T = 3 tokens
C = 2 ( embedding dim )
$d_k = d_v = 2$

these $d_q$, $d_k$, and $d_v$ are the dimensions the query, keys and values live in
this is def meant to be $\le C$ else you are "creating" more information
from less which is just wrong

Say we have three tokens in input -> $x_0$, $x_1$, and $x_2$
to use those we use their embeddings $\operatorname{emb}(x_0)$, $\operatorname{emb}(x_1)$, $\operatorname{emb}(x_2)$

q, k, v operate in a smaller dimension usually so we do a projection,
lossy if the dimensions are less, using $W_q$, $W_k$ and $W_v$ as projection
matrices, note that we learn these $W$ projection matrices

$$
x_0 = [1,0], \quad x_1 = [0,1], \quad x_2 = [1,1]
$$

$$
W_q = \begin{bmatrix}1&1\\0&1\end{bmatrix}, \quad
W_k = \begin{bmatrix}1&0\\1&1\end{bmatrix}, \quad
W_v = \begin{bmatrix}1&0\\0&2\end{bmatrix}
$$

Project each token in q, k and v dimension each

Query -> what I'm asking for
Key -> what I use to match against with that query
Value -> once matched, what I use

$\operatorname{Score}(i,j) = q_i \cdot k_j$
how well this query i matches against key j
magic for math reasons -> $S(i,j) \mathrel{/}= \sqrt{d_k}$
Mask where query is matching against future ( $j > i$ ) = -inf
what is each row now ?
score(i, ...j ) -> for each query i, including only past j
how well is the match as a "logit"
softmax the row to get to preds ( the idea is to normalise and clamp to
1 otherwise a 10 on one row does not compare at all to a 40 on another row
or a 500 on some other )
A - these preds we got post softmax cool did we push it?
= attention matrix

note that this A is a lower left triangle, upper half is masked

now $\mathrm{out}_i$ => for each query token weight by probabilities the values ( we USE
values )

$$\mathrm{out}_i = \sum_j A(i,j) \cdot v_j$$

match score probability for jth key $\times$ jth value
so then this output is $d_v$ dimensional

in practice "head_size" = $d_v = d_k$
> there is no separate $d_q$, since we need to dot $d_k$ MUST = $d_q$

### Position embedding

all this is still a bag of words, there is no position
so what we do is "ADD" ( not concat ) a position embedding
a concat would be ideal really but we see that high dimensional vectors are
in general mostly orthogonal ( close to ) so addition is fine

why not just append an index based token?
*todo: this I should try* but the general idea is that there is not enough
"non-linearity" to it, even though it goes through a whole W_q projection

### Why higher dims almost orthogonal

$$\text{dot product} = \sum_i u_i v_i$$

ui and vi are independent, their products then are too, idea being you start at
0 and accumulate deltas = $u_i v_i$ which is similar to a random walk of len d
so as there are more terms it's closer to 0
=> num grows as $\sqrt d$ ( random walk grows as $\sqrt d$ ) and when you norm
that vec the scalar becomes $1 / \sqrt d$

so in input we get

$$x = \mathrm{token\_emb} + \mathrm{pos\_emb}$$

( not concat )

### Actual attention model

key, query and value all becomes linear transformations with no bias
( older gpt models used bias but it was empirically found that there is almost
0 loss on dropping them; some argument being that softmax is "shift" invariant
so constant and same bias term multiplied across the row make no change
effectively; there is a term still left and not proven mathematically but yeah
empirically good ).

### Generation with attention

the same training loop lets us be slightly better than a bigram though since
we are still just doing it char level over one head, there are limits

note that A is the weight assigned to each position, not actually probabs
of the next token

in fact, note that the cell does not even use v or the further layers at all

## Model architecture

### To logits (LM head as it’s called in BERT and incorrectly named as such in some implementations)

this is used as a projection to "vocab_size" post attention stuff
in most of ml you keep on reducing dimensions collecting higher and higher
features but in here, for llms, this goes the other way around, the token /
vocab size is much much larger than embed dim, so apply softmax you need
a logit for each "token" / "class", number of classes dominate way more
and hence you do a linear mapping from what is a lower dimension to a higher
dimension with entries being dependent -> t -> (2t, 3t+10, 8.6t)

### Low rank bigram

so far what we have is x = ( tok + pos embds )
embd being lower than full rank and then we project it to full rank

because of the rank loss I would say it's weaker bigram

the pos values also contribute nothing, they say char c comes at positions 3
but since I random sample with no regard for boundaries it could've just been
in any other position based on where we randomly split off at.

### To multi head

we need a non linearity
( todo ) on the details as to why but rn effectively multiple linear transforms
are sequential so it's effectively just one

### FFN / MLP

FFN is just an MLP, it's FFN because the og transformer paper called it so
no point in re-inventing new names

( todo ) non linearity

a perceptron is just activation( lin transform )

mlp is just a linear transform after this
so lin( act( lin ) )

```text
what's "layer" ?
h0 = act( W x + b )
hi = act( W h(i-1) + b)
y = W h(n-1) + b   // no act -> readout
```

### Attention one liner

$$\mathrm{logits} = W_{\mathrm{lm}} \cdot \left[\operatorname{softmax}\!\left(\frac{(W_q x)(W_k x)^T}{\sqrt d} + \mathrm{mask}\right) \cdot (W_v x)\right]$$

```text
|       |          |---------------- A(x), (T,T) --------|   |
|       readout                 routing weights              values
```

$$x = E_{\mathrm{tokens}} + P_{\mathrm{positions}} \qquad (B,T,C)$$

Note that everything after A softmax is one chain or matrices W_v and W_lm
the linear transforms collapse to just one
this is where we put in FFN we learn the "curves" in between

$$\mathrm{logits} = W_{\mathrm{lm}} \cdot W_2 \cdot \operatorname{ReLU}\!\left(W_1 \cdot [A(x) \cdot W_v x] + b_1\right) + b_2$$

```text
| | | | | | | | | | |--------- the only bend ---------|
```

A learns who is relevant to what
$W_v$ learns its contribution | given relevance

what is input to FFN? ( B T C )
for each batch, we see for each token, it's A V summary ( A is what weight based on key query relevance and V being the
actual contribution )

FFN only applies across the last dim, and mixes the C

### Actual FFN

here this will just be an mlp with ReLU

this is just one layer deep, one relu
dims of the middle layer = counts of relu = count of non-linearities in there ( hinges )
ideally we want more of such bends but can't just increase C ( n_embd ) so empirically increase
where cheap, 4 is what the og paper used

### Splitting attention

a single attention head learns W_q,k,v off of random init, based on where it started off randomly
it can learn different features like previous character's value, characters at boundaries etc
but ONE softmax has to commit to one so using just ends up less than ideal
in practice we split it into multiple heads, have each one learn "possibly" something diff and then
concat, generally research shows most heads learn nothing useful though multiple is better as some
do

a head pruning seems a bit under-developed idea to me but well you cannot really split it into too many heads
what is split? usual multi head splits dims and assigns one head to each; you can copy as well and this was
measured in the og paper

the common MultiHeadAttention
you fix n_heads = number of heads

attention still does not modify dimensions

`head_size = n_embd/n_heads` # must be divisible
at last concat them to get back to n_embd

there is one more step done here in practice => another linear projection Wo

note that with just this I'll be at concat -> Wo -> FFN -> l1 -> act -> l2
that Wo and l1 collapse so no gain here from a Wo
=> impl later, residual connections after Wo will fix that

( todo ) rank ceiling issue if you make n_heads too high

### Stacking “blocks” to go deep

block = mha + ffn ( right now my mha is separate module but ffn is in the combined model )

residual stream = Output after ffn
```text
x0 = embed + pos
x1 = mha(x0)
x2 = ffn(x1)
logits = W_(lm_head I think I renamed it to something but yeah we want logit dim to be vocab size, this'll do a proj) . x2
```

> showing we need multiple layers ( INCOMPLETE )
paper from elhage that with an experiment shows 2 layers can learn what one layer just cannot
this x2 is the "residual" for one layer
note that we are talking about multi layers here, so residual0 = tok + pos emb
residual1 l1 is the ffn
layer 1 does attention with keys derived only from position

```text
pos:  0   1   2   3   4
tok: [A] [B] [C] [D] [A]  -> predict [B]
```

trigram => AB : predict C = next
skip-trigram => gap bw A and B is arbitrary => A...B : predict C = next of B
not running anything here, argumentative is good enough
pos 4 A can ( query ) can learn to match with key of pos 0 A or pos B 1 and then
learn that value for that match position ( whichever it attends to ) is B
note that NONE of this comes from context, but from weight
say it learned to match A with A and that B follows A
CC is just hallucinative, we have reached no coherent result, dropping this

> offtopic: do we not use dropout on these at all
residual are not dropouts, was mixing this up
in general for llms we don't use dropouts because there the idea is of regularisation
needed due to possible over-fitting due to seeing the same data again and again
here text corpus is so big that most of the data is never seen in a random training
so block = mh + ff

at layer 6 this was way bad
( todo) check this bit about some rank collapse
> CC: This is the rank collapse from Dong et al., and it bites at depth 3, not at depth 12. I was wrong about the scale.

running at layers = 2 it converges better but does not really improve over baselines in any significant way
I believe it's just more params, something about signal being too "mixed" up when going deeper to predict anything
reasonable, or maybe the magnitudes diminish? ( todo, again )

depth without residuals does not help and instead makes it worse

### Residuals

> offtopic: layernorm is not regularisation, I somehow got this idea that it is, batch norm is, somewhat at least

currently we have multiple layers of

```text
block(x):
| x = mha(x)
| x = ffn(x)
```

at each block only what block at that layer learns to keep remains so not only does it need to learn to keep relevant bits
BUT also learn new relations, so overall they contribute nothing, I guess all they learn is to "keep" the same signal that
the original one found or learn something tangential from it and OVERWRITE it
also attention as convex combination ( $\sum_i p_i$, $p_i \ge 0$, so just like probabs ) so it does a "weighted" smoothing across the
values, each block does this "smoothing" eventually the signal gets too smoothed out as we go deep

idea of residuals is that instead of overwriting, we add, so instead of signal 2 = f( signal 1 )
we do signal 1 + signal 2

reminder on ffn and attention
ffn acts across C after attention has already blended values FOR that position
it just "summarises" that value blend via a linear layer

two residuals or one?
now with this, you can add residual AFTER the whole block or add it once after both mha and ffn

if you just do one, then x is just ffn so ffn needs to learn to BOTH summarise and preserve
==> once again,
attention mixes across positions, given you a convex combination
ffn makes hinges across C so learn complex features
==> don't understand this well enough still but ( todo )

currently we have, x = mha = attn -> linear on x
then an ffn on x = lin -> relu -> lin

currently the last linear on mha and the first one on ffn are collapsed, idea was we add residuals here
so when we add that the input to ffn is different from the output of mha so not collapsed anymore
but that is secondary, a bit atleast

CC just says -> "giving direct line to stream" and seems all transformers do it this way

anyways, we move on

### Layer norms

$L$ = number of "sublayers" = what all contribute an "additive" weight to x as the model goes
this here is $2 \times$ blocks as each block has two, mha and ffn

the problem is that each sublayer "accumulates" signal

the idea of residuals was that we want to accumulate signals, accumulating magnitudes are a side effect
problem from that cause

1) larger magnitudes increase the effective learning rate for deeper layers
which then have larger gradients due to the higher magnitudes so higher
layers end up over-adjusting

2) softmax saturation => softmax is not scale invariant

```text
|scores [0, 1, 2]        -> [0.09, 0.24, 0.67]    soft blend
|scores [0, 12, 24]      -> [~0, ~0, 1.00]       one-hot
```

===> prereq => change in mag and direction of sum of high dimensional vectors

$$x_L = \sum_i x_i$$

where each $x_i$ is in $\mathbb R^n$

if any two vecs are ALMOST orthogonal
dot product is a sum of random walks, so $u \cdot v \to 0$
simply $\mathbb E[u \cdot v] \to 0$ and $\operatorname{std}[u \cdot v] = 1 / \sqrt C$ ( todo: derive )
so as the dimension increases it's more and more closer to 0
cosine being 90 degrees then

> $$\|a+b\|^2 = \|a\|^2 + \|b\|^2 + 2\langle a,b\rangle$$
idea being that same as scalars, dot product as a bin op is same as scalar mult
( commutative and distributive )

so that dot product drops out as -> 0
say we have k of those additions, each of len w

$$\|x_L\| = \|w\|\sqrt{k}$$

grows in $\sqrt k$ => k is the number of times we add

--- whatever I forgot exactly but there was 1 more 3rd point here
