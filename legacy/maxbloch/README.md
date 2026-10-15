# maxbloch → mbsolver

This package has been renamed to **[mbsolver](https://pypi.org/project/mbsolver/)**
and gets no further updates. Install the new name and change the imports:

```
pip install mbsolver
```

```python
from mbsolver import Pulse, Steps          # was: from maxbloch import ...
from mbsolver.two_level import solve_1d    # was: from maxbloch.two_level import ...
```

Nothing else changes. Version 0.1.1 is the same code as 0.1.0 and only adds a
deprecation warning on import. (It does not install `mbsolver` itself, because
the old Maxwell Bloch Solver app used the import name `mbsolver` for itself.)
