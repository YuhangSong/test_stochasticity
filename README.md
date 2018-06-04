# Test Stochasticity of ALE

Code for testing the stochasticity of ALE.

### Requirements

* [numpy](http://www.numpy.org/)
* [atari-py](https://github.com/openai/atari-py)
* [Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)

### Run the code

To test the stochasticity of loadROM, set ```test = 'loadROM'``` in python file ```test_stochastic.py```, and then run ```python test_stochastic.py```.
It should produce:
```
game:pong grouped_num:1 distribution:[200]
game:assault grouped_num:57 distribution:[4, 4, 6, 3, 8, 9, 2, 5, 3, 2, 5, 5, 5, 4, 5, 4, 2, 4, 6, 3, 1, 4, 1, 5, 5, 1, 2, 1, 3, 6, 3, 4, 2, 3, 5, 3, 4, 6, 3, 5, 4, 3, 2, 2, 3, 4, 2, 1, 3, 4, 3, 1, 2, 2, 5, 2, 1]
game:ms_pacman grouped_num:1 distribution:[200]
```

To test the stochasticity of restoreState, set ```test = 'restoreState'``` in python file ```test_stochastic.py```, and then run ```python test_stochastic.py```.
It should produce:
```
game:pong grouped_num:1 distribution:[200]
game:assault grouped_num:1 distribution:[200]
game:ms_pacman grouped_num:1 distribution:[200]
```
