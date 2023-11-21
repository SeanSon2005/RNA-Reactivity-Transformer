from arnie.mfe import mfe
import numpy as np

sequence ="AUAGCUCAGUGGUAGAGCA"
structure = mfe(sequence,package="eternafold")
print(structure)

#export ETERNAFOLD_PATH=/home/sean/Documents/Coding/RNA/EternaFold/src
#export ETERNAFOLD_PARAMETERS=/home/sean/Documents/Coding/RNA/EternaFold/parameters/EternaFoldParams.v1