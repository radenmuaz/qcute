Lots of alternatives
768 ÷ 128 = 6 → not below 6
768 ÷ 256 = 3
768 ÷ 64 ÷ 2 = 6 → not below 6
768 ÷ 64 ÷ 4 = 3
768 ÷ 32 ÷ 4 = 6 → not below 6
768 ÷ 32 ÷ 8 = 3
768 ÷ 16 ÷ 16 = 3
768 ÷ 16 ÷ 8 ÷ 2 = 3
768 ÷ 16 ÷ 4 ÷ 4 = 3
768 ÷ 16 ÷ 2 ÷ 2 ÷ 2 = 6 → not below 6
768 ÷ 8 ÷ 8 ÷ 4 = 3
768 ÷ 8 ÷ 4 ÷ 4 ÷ 2 = 3
768 ÷ 8 ÷ 2 ÷ 2 ÷ 2 ÷ 2 = 6 → not below 6
768 ÷ 4 ÷ 4 ÷ 4 ÷ 4 = 3
768 ÷ 4 ÷ 4 ÷ 4 ÷ 2 ÷ 2 = 3
768 ÷ 2 ÷ 2 ÷ 2 ÷ 2 ÷ 2 ÷ 2 ÷ 2 = 3


Yes. Assuming we're talking about the **flattened sequence**:

 $$
64\times64\times3=12,288
$$

 and you want the final length to be **16**, the total downsampling factor must be:

 $$
\frac{12,288}{16}=768=3\times256=3\times2^8
$$

 With the constraint that **at least one factor must be 3**, here are good combinations:

 ### Simple combinations

 - `[3, 256]` → $3\times256=768$
- `[3, 128, 2]` → $3\times128\times2=768$
- `[3, 64, 4]` → $3\times64\times4=768$
- `[3, 32, 8]` → $3\times32\times8=768$
- `[3, 16, 16]` → $3\times16\times16=768$
- `[3, 8, 32]` → $3\times8\times32=768$
- `[3, 4, 64]` → $3\times4\times64=768$
- `[3, 2, 128]` → $3\times2\times128=768$

 ### If you want several layers with relatively small factors

 These are more practical:

 - `[3, 4, 4, 16]`
- `[3, 4, 8, 8]`
- `[3, 2, 4, 32]`
- `[3, 2, 8, 16]`
- `[3, 2, 2, 8, 8]`
- `[3, 2, 2, 4, 16]`
- `[3, 2, 2, 2, 4, 8]`
- `[3, 2, 2, 2, 2, 4, 4]`
- `[3, 2, 2, 2, 2, 2, 2, 8]`

 All of these satisfy:

 $$
\boxed{\prod_i x_i=768}
$$

 so:

 $$
12,288 \rightarrow 16
$$

 For example, with **`[3, 4, 8, 8]`**:

 $$
12288\xrightarrow{/3}4096
\xrightarrow{/4}1024
\xrightarrow{/8}128
\xrightarrow{/8}16
$$

 **My preference:** `[3, 4, 8, 8]` if you want 4 downsampling layers, or `[3, 2, 4, 32]` if you want fewer layers.