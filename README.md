# Physics-Simulations
Repo containing various physics simulations in python

## Getting Started
```console
pip install -r requirements.txt
```

## Waves 3d  
**Run:**
(It needs to be run as a module (`-m` flag) to be able to access the [`colors.py`](colors.py) file 
```console
python -m waves3d
```
**Movement:**
  - Use the arrow keys to move around, scroll to zoom in or drag to rotate. With <kbd>Ctrl</kbd> pressed
and the arrow keys you can rotate too.
  - Press <kbd>r</kbd> to reset to the initial starting position

**Customization:**      
  - Change between the different grid types in the `Simulation` constructor. Use the function `cube` that takes in 
a grid and a he9ight to make the grid cubical. It returns a new class that you can use top construct the wave as with any other.
  - Modify constants scattered around to see different outputs
  - Modify [`waves3d/wave.config`](waves3d/wave.config) to change the waves. You can dop this dinamically and press <kbd>F5</kbd> 
    to refresh the simulation. 
  - Define constants: `#define <name> <value>` value must be float  
  - Comments start with `//`.
  - A wave is defined with 4 float parameters: Amplitude, Periodm, Wave Length and Phase separated by spaces
  
**Demo video**: [demo.mp4](https://user-images.githubusercontent.com/72514269/151450606-4a5aecd2-4f89-45d2-a564-e09310d22032.mp4)  
**References**:  
Python and OpenGL  online book: https://www.labri.fr/perso/nrougier/python-opengl/#modern-opengl

