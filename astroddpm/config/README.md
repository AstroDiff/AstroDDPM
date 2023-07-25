# Use of the config folder

The config folder has two main uses:

* to store flagfile/argfile for efficiently passing argument to the command line
* to serve as backup of models trained and experiments conducted.

## Flagfile

Standard absl flag flagfile syntax. See DDPM_1.02 for an example. Work in progress to use argparse.

## Experiments backcup

After parsing arguments, the app immediatly store them in a flagfile in the model own folder inside the checkpoints directory. To access it you need to know where it is stored. This is why for automatic sbatch experiments where no flagfile is called, a copy of the arguments used is stored here under a file named exp[...].txt. You can then check the flagfile in either the specified ckpt folder (or the default one if none was specified) to have a complete list of all the arguments (including default values)
