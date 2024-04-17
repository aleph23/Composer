# Composer
Generate tunable music using neural networks.

Repository to ["Generating Songs With Neural Networks (Neural Composer)"](https://youtu.be/UWxfnNXlVy8).

## How to install
* Requires Python 3.5.6
* CUDA 10.0 AND CUDNN 7.6.5 https://developer.nvidia.com/rdp/cudnn-archive
* Install dependencies in python3 by running `pip install -r requirements.txt`.

## How to run

* Find some dataset to train on. More info on where to find datasets can be found in `data/raw/README.md`. A default bach dataset is included in the repository.
* Run `preprocess_songs.py`. This will load all midi files from your midi files folder `data/raw/` into `data/interim/samples.npy` & `lengths.npy`.
  You can point the script to a location using the `--data_folder` flag one or multiple times to make it include more paths to index midi files.
* Run `train.py`. This will train your network and store the progress from time to time (EPOCHS_TO_SAVE to change storing frequency).
  Only stops if you interrupt it or after 2000 epochs. You need to change "CONTINUE_TRAIN" to False to start training from scratch.
* Run `composer.py --model_path "e300"` where `"e300"` indicates the folder the stored model is in. Look into the `results/history` folder to find your trained model folders.

## Composer

The Composer script will load your model trained on the midi files you provided
into the music generator.
Internally, the model tries to compress a song into only 120 numbers called the latent space.
Seen from the other perspective, the latent space numbers encode each song that the model
is trained on. Changing the latent numbers allows us to sample new, unknown songs.
To make the sampling more efficient, we use PCA, a way to extract the dimensions in which there is 
most variations across all songs, to make the influence on the new songs strong. 
The sliders in the composer adjust those numbers and are ordered from most
important/influential (left) to least important/influential (right).  Just the
top 40 are shown on screen.  For more details about how
this works and other fun projects, please check out my
YouTube channel 'CodeParade'.  Have fun making music!

=========================================================
```
Composer Controls:
 Right Click - Reset all sliders
         'R' - Random Song
	 'E' - Random Song 2 stdev
	 'M' - Save song as .mid file
         'W' - Save song as .wav file
	 'O' - opens the loaded samples and plays the reconstructed version of one of the input songs
         'C' - Does a thing, no idea how it works or what it does
	 'Escape' - Quit
	 'Space' - Play/Pause
         'Tab' - Seek to start of song
         '1' - Square wave instrument
         '2' - Sawtooth wave instrument
         '3' - Triangle wave instrument
         '4' - Sine wave instrument
	 Red Slider - Threshold to play a note
	 Green Slider - Speed of song
	 Blue Slider - Volume
	
Exclusive to this fork
	 Yellow Slider - Note Duration
	 Cyan Slider - Note Balance (Makes low notes louder than high notes or vise versa)
	 '5' - Circle wave instrument
         'T' - Randomizes last 2/3 of sliders
         'X' - Randomly alters each slider (add a random number 0.1 stdev to the current value)
         ',' - Multiplies each slider by 1.1 (I chose comma because on my keyboard it is the lowercase of <)
	 '.' - Divides each slider by 1.1 (I chose period because on my keyboard it is the lowercase of >)
	 '/' - Multiplies each slider by -1
	 Up Arrow - increments the note threshold (Useful for trying to add one note at a time)
	 Down Arrow - decrements the note threshold (Useful for trying to remove one note at a time)
         'S' - Save slider values to text file (includes threshold, speed and volume sliders, model path and instrument number)
			Asks in command line what filename to save as
			To load a saved song run the command >py composer.py --model_path song1.txt
			It knows to load the saved song file because of the .txt extension and gets the model path from inside the file
			If blend mode is on, it will save a file containing the names of the blended songs that can be opened in the same ways
	 'L' - Loads a song
			Asks in command line what song file (created by the S command above) to open and play without closing the app
         'A' - Toggles auto save whenever the song finishes, it is saved and a new song is randomly generated.
			Could be useful if you are listening while doing something else and want to look back on the history.
	 'B' - Blends smoothly through a series of preset songs by taking a linear combination of the latent vectors. Sounds awesome.
			asks in command line what song files (created by the S command above) to blend
			Example result in blended song.mp3 which was recorded with audacity
			https://drive.google.com/file/d/17MNTsHMXghApAa_GcUMB-pY0PTWWRnIF/view?usp=sharing

I would have liked the text entry to be in a box to the right of the notes rather than in command line, but I couldn't figure out how to do that
If you can help with that please make a pull request and email me at
michael_einhorn@yahoo.com
	 
```

## Pretrained models

Below are some pretrained models for different datasets. To use the model, download the zip from the
link below and extract it into the results/history folder of your project. The zip should contain a folder 
named e and some number, indicating the number of epochs the model was trained and some model.h5.
Pay attention when extracting the model not to override one of your own trained, most beloved models!

* Bach dataset: https://drive.google.com/open?id=1P_hOF0v55m1Snzvri5OBVP5yy099Lzgx
