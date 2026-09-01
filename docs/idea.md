new task
summformer_jax/image_gen_blockdecode
sketch first something like qcute lagcodec causal (lag=1) and fractal ar
given 64x64 image
assume image given as patch of 16x16 or single 64-dim scanline but as tokens across 64 timesteps
also now summformer instead of single model trunk with side code lm, 
function as encoder decoder style like, codelm read source directly not from trunk

format can either square patch or lines (64x64-> 64-line-> 16-line -> 4-line -> 1 pixel rgb)

encoder does codelm thing, given 64x3 rgb -timesteps input, downsample 3 stage,
64, stride 4, to 16, stride 4, to 4, stride 4, to 1
let this level, timesteps count:

0, 64
1, 16
2, 4
3, 1

each stride last timestep hidden state, use them to cond to feed to self attention decoder

another design decision not using cross attn fusestage, decoder simple self attn
decoder is block parallel causal, prepend cond to attn with encoder outs above
goal is to predict next 64x3 rgb timestep, only 3 timestep
there will be basically batched 64 paralle decodes
each 64 has different conditioning but partial sharing,
simple self attention concat prepend as embedding input then predict rgb

e.g. recursive several level cross attn before rgb head
1 64-group cond level 3,
4 16-group cond level 2,
16 4-group cond level 1,
give one trainable query token to ntp against 3-step rgb targets


