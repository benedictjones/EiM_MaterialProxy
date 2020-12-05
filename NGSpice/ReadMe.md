# Copy Of NGSpice

A copy of the NGSpice version used on windows is included.
Unzipping NGSpice.zip will provide you with 2 files: Spice64 and Spice_dll.

## Instillation
Note, that depending on the PySpice version this differs.
Detailed Instructions of instillation are given on https://pyspice.fabrice-salvaire.fr/releases/v1.4/ 

- For PySpice 1.3: Place Spice64 and Spice_dll in C:\Program Files

- For PySpice 1.4: Place Spice64 and Spice_dll in C:\Users\user\anaconda3\Lib\site-packages\PySpice\Spice\NgSpice

## Known Bugs

While using PySpice/NGSpice I found that a Yaml *error suppression* warning is thrown.
 
To prevent the Yaml displaying a warning when running PySpice. Go to *C:\Users\user\Anaconda3\Lib\site-packages\PySpice\Logging\Logging.py* and change *yaml.load()* to *yaml.safe_load()*.

Note: this issue seemes to be fixed now.