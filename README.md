## Transferable adversarial attack on image tampering localization
An official implementation code for the paper "[Transferable adversarial attack on image tampering localization](./paper.pdf)" published on JVCIR 2024.

### Proposed Scheme
![proposed_network](./pit.png)

Figure (a) Overview of the proposed adversarial attack framework. (b)(c) Proposed optimization-based and gradient-based attack methods against image tampering
localization algorithms, respectively.

### Usage
preparation:
```python
python get_label.py
```

optimization-based attack:
```python
python opt.py
```
	
gradient-based attack:
```python
python grad.py
```

## Bibtex
 ```
@article{cao2024AttackITL,
 title={Transferable Adversarial Attack on Image Tampering Localization},
 author={Gang Cao, Yuqi Wang, Haochen Zhu, Zijie Lou, Lifang Yu},
 journal={Journal of Visual Communication and Image Representation},
 year={2024},
 publisher={Elsevier}
}
 ```
## Contact
If you have any questions, please contact me(wangyq0920@163.com).
