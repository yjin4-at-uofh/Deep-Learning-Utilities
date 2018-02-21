# Deep Learning Utilities

*****
```
 __, _,    __, _, _ _,_  _, _, _  _, _ _, _  _,   ___  _,  _, _,   _,
 | \ |     |_  |\ | |_| /_\ |\ | / ` | |\ | / _    |  / \ / \ |   (_ 
 |_/ | ,   |   | \| | | | | | \| \ , | | \| \ /    |  \ / \ / | , , )
 ~   ~~~   ~~~ ~  ~ ~ ~ ~ ~ ~  ~  ~  ~ ~  ~  ~     ~   ~   ~  ~~~  ~ 
```
*****

## Yuchen's Deep Learning Enhancing Tools - Readme

This is a collection of deep learning utilities. You could use it to pre-process some data and do something on numpy arrays efficiently.. Just use it as a common module in python like this.

```python
  import dlUtilities as du
```

Noted that this API need you to install numpy. 

## Usage

### Matrix Projecting

Suppose that we have a matrix that has `[height, width, ..., channels]`, and we need to project the dimension `channels` into another axis so that we would get a matrix like `[height, width, ..., newchannels]`. Then we may need a dictionary to help us complete this process. For example,  if the input matrix has 3 channels, and we may want to project it into 5 channels. We may use such a projection dictionary:

```python
proj_dict = {
    [0  ,128,0  ]:[0,0,0,0,1],
    [128,0  ,0  ]:[0,0,0,1,0],
    [0  ,0  ,128]:[0,0,1,0,0],
    [0  ,64 ,64 ]:[0,1,0,0,0],
    [64 ,0, ,64 ]:[1,0,0,0,0]
}
```

which means in the channel axis, we could only have 5 possible kinds of values when considering the data distributions. So we could project it into an one-hot matrix with 5 channels. Noted that a dictionary like the above one is not available, because python does not support you to hash a list-like object. Therefore we use a list as a substitute as our codes:

```python
proj_dictList = [
    [np.array([0  ,128,0  ], dtype=np.uint8), np.array([0,0,0,0,1], dtype=np.uint8)],
    [np.array([128,0  ,0  ], dtype=np.uint8), np.array([0,0,0,1,0], dtype=np.uint8)],
    [np.array([0  ,0  ,128], dtype=np.uint8), np.array([0,0,1,0,0], dtype=np.uint8)],
    [np.array([0  ,64 ,64 ], dtype=np.uint8), np.array([0,1,0,0,0], dtype=np.uint8)],
    [np.array([64 ,0, ,64 ], dtype=np.uint8), np.array([1,0,0,0,0], dtype=np.uint8)]
]
```

Use the following code to do this work:

```python
  pjc = du.Projector() # Create projector
  pjc.registerMap(proj_dictList) # Register the transfer list
  output = pjc.action(input) # Project the matrix
  rec_input = x.action(output, True) # Recover the matrix from the output of forward projection
```

We would find that this handle could project the matrix and recover it with the same dictionary.

### I/O Data
This handle also support you to read some special data by high-level API with low-level realization. Now we only support 1 mode.

#### Read Seismic Data
This is an API for reading the raw data from shot/receiver collections. Use the following code to get the avaliable data.

```python
    dio = du.DataIO() # Create the handle
    dio.load(b'path',b'seismic') # Load source files
    print(dio) # Show avaliable information
    print(dio.size()) # Get information: [shot number, receiver number, time step]
    p = dio.read([1, 30]) # Read the 1st and the 30th data collection from the corresponding shots
    print(p.shape, p.dtype)
    print(x.size())
```

For more instructions, you could tap `help(du)`. 

## Update Report

### V0.5 update report:
1. Provide the `Projector` tool and `DataIO` tool.
 
## Version of currently used Python library
* Python 3.5
* numpy 1.13
