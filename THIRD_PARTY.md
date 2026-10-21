The integration uses the following upstream projects without vendoring their code into git:

- [NVlabs/RADIO](https://github.com/NVlabs/RADIO), commit
  `4ffaf15d0ff48197c194e306772bdb5fb92db230`.
  C-RADIO weights have their own NVIDIA license; consult the
  [C-RADIOv4-H model card](https://huggingface.co/nvidia/C-RADIOv4-H).
- [microsoft/MoGe](https://github.com/microsoft/MoGe), commit
  `74fbce054ebed49800de42d0ad0e83495065719a`, MIT license,
  Copyright (c) Microsoft Corporation.
  `decoder.py` follows the v2 decoder ordering and delegates inference
  postprocessing to this pinned upstream source.
- [utils3d-moge](https://github.com/EasternJournalist/utils3d-moge), commit
  `62f09d58509485564e24d5d9f6aac9ee9ebc0c37`.

The upstream MoGe license is reproduced below for its decoder structure:

MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
