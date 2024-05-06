# yelp-restaurant-recommender
Using open-clip to power a recommendation system based on image classification through simultaneous textual and visual embedding of captioned Yelp review images.

## Getting started
This project is a WIP that is currently undergoing more organization for reproduceability. Using a HPC environment with GPU processors is greately encouraged.

### Prerequisites
A function python version of 3.11 is recommended for ensuring compatibility. 

### Installation
WIP

## Usage
The full dataset with images can be downloaded on [Yelp](https://www.yelp.com/dataset), the ```Dataset_User_Agreement``` file contains guidelines and restrictions for how this dataset should be used. 

The downloaded data should be extracted (about 10GB) and placed into a ```photos``` folder.

Project Structure:  
*disclaimer: There might be path discrepencies due to difference in HPC file organization*

├──feature_extraction # For pre-processing of data and feature extraction using CLIP
├──fine_tuning # For fine tuning and transfer learning of CLIP with the help of CoCa
├──clustering # For clustering of feature embeddings and resulting analysis  
├──out # For model outputs  
├──test # For storing a separate test used for eval


