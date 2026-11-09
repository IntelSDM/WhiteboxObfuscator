# Whitebox Obfuscator
Built for LLVM 19.1 - Windows x64 <br> <br>

Generates a unique whitebox for encrypting constants. <br>
This is more or less a mashup of the past 20 years of whitebox development into an LLVM obfuscator for constant encryption. <br>

| Axis | UniqueValue |
|---|---|
| **Block size** | `{8, 16}` bytes | 
| **Round count** | `{6, 7, 8}` |
| **MDS family** | `Cauchy` / `Vandermonde` / `Hadamard` |
| **ShiftRows kind** | `Random` / `AESAnalog` / `BitRotBaked` |
| **S-box scheme** | `Shared` / `PerRoundIndependent` | 
| **Round program** | `Canonical / OrderSwap / NoARK / LinearOnly / ParallelSBoxes` per round |
| **Fusion granularity** | `Full` / `ARKOut` per Canonical round |
| **External encoding** | `count ∈ [1..8]`, `depth ∈ [1..4]`, `family_size ∈ {64,128,256,512,1024}` | 
<br>
In my honest opinion, outside of forms of packing and licensing, I simply can't find much of a justification for whiteboxes within modern code obfuscation. <br> 
So just to save you time if you are going to jump down this rabbit hole; as I did. I am yet to find a usecase unless you want to do this in JIT and pair them to an actual per VM instance, then I could definitely see a usecase, but you will use 10mb of ram for very little gain. <br>
I will say the only real usecase for this is if you want to store VM Opcodes in a secure format until they are mapped after unpacking. <br>
<br>
<img width="1977" height="575" alt="image" src="https://github.com/user-attachments/assets/3d3131b5-c36b-4baf-af4b-ca10b3c7a53f" />
<br>
<img width="1963" height="557" alt="image" src="https://github.com/user-attachments/assets/2406b967-d460-41c7-aa68-d6e41ffe6e9a" />


## Academic References

[White-Box Cryptography and an AES Implementation](https://home.cs.colorado.edu/~jrblack/class/csci7000/f03/papers/oorschot-whitebox.pdf) <br>
[A White-Box DES Implementation for DRM Applications](https://crypto.stanford.edu/DRM2002/whitebox.pdf) <br>
[Cryptanalysis of a White Box AES Implementation](https://link.springer.com/content/pdf/10.1007/978-3-540-30564-4_16.pdf) <br>
[Cryptanalysis of a Generic Class of White-Box Implementations](https://link.springer.com/content/pdf/10.1007/978-3-642-04159-4_27.pdf) <br>
[Cryptanalysis of the Xiao–Lai White-Box AES Implementation](https://link.springer.com/content/pdf/10.1007/978-3-642-35999-6_3.pdf) <br>
[Two Attacks on a White-Box AES Implementation](https://link.springer.com/chapter/10.1007/978-3-662-43414-7_14)
[White-Box AES Implementation Revisited](https://eprint.iacr.org/2014/688)
[Differential Computation Analysis: Hiding Your White-Box Designs Is Not Enough](https://eprint.iacr.org/2015/753)
[Unboxing the White-Box: Practical Attacks Against Obfuscated Ciphers](https://blackhat.com/docs/eu-15/materials/eu-15-Sanfelix-Unboxing-The-White-Box-Practical-Attacks-Against-Obfuscated-Ciphers-wp.pdf)
[Protecting White-Box AES with Dual Ciphers](https://link.springer.com/content/pdf/10.1007/978-3-642-24209-0_19.pdf)
[A Secure Implementation of White-Box AES](https://eprint.iacr.org/2013/104.pdf)
[Implicit White-Box Implementations: White-Boxing ARX Ciphers](https://eprint.iacr.org/2022/428)
[Cryptanalysis of ARX-based White-Box Implementations](https://tches.iacr.org/index.php/TCHES/article/view/10958/10265)
[The Advanced Encryption Standard](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.197-upd1.pdf)
[The design of linear transformations for substitution permutation encryption networks](https://dl.acm.org/doi/pdf/10.1145/191177.191206)
[Better Bit Mixing - Improving on MurmurHash3's 64-bit Finalizer](https://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html)
[Attacks and Countermeasures for White-Box Designs](https://eprint.iacr.org/2018/049.pdf)

## Annotations
```c
__attribute__((annotate("obfus")))
``` 
* Encrypts Constants With Whitebox Table

## How To Build
* Download [LLVM 19.1](https://github.com/llvm/llvm-project/releases/tag/llvmorg-19.1.0)
* Download [VCPKG](https://github.com/microsoft/vcpkg)
* VCPKG install asmjit
* Open build.bat in a text editor.
* Find the LLVM 19.1 references in the command line, Change them to your file path.
* Find the VCPKG references in the command line, Change them to your file path.
* Run the build script
* Build your application with the ``` -load-pass-plugin ``` Param to the FunctionPass.dll
