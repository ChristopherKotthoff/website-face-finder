# website-face-finder
Find a specific face by similarity search to an input image over a given website domain. The tool works by doing some scraping and then some image analyzing alternatingly. It scrapes the website in a breadth-first manner, collecting image-urls and further links pointing to the same website. Then, the the images are iterated over to find similar faces to the input image.

Similarity search is done with a pre-trained lightweight model called ['buffalo_sc' from insightface](https://github.com/deepinsight/insightface/tree/master/python-package) which uses [onnxruntime](https://onnxruntime.ai/docs/get-started/with-python.html) in the backend. CPU is quite doable. Model can be changed to to a better one via argument but is not needed.

Inference can be done in CPU or GPU, depending on what you have and which onnxruntime version you install


# Installation
Install g++, it is needed by insightface. On ubuntu:
```bash
sudo apt-get install g++
```

Install the pip packages:
```bash
pip install -r requirements.txt
```
or 
```bash
pip install -r requirements-gpu.txt
```
depending on if you want to use GPU or not. GPU version also works on CPU if no GPU is available.
If you have CUDA version 11.8 instead of 12+, you can install onnxruntime in the requirements-gpu.txt file like this:
```bash
onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
```

I have had an error where np.int is not supportet anymore because of a new numpy version. If you try to run the program and you see something about np.int, go to the insightface library file that it complains about and replace all occurances of np.int with int.

# Reminder
Only use on sites that you are allowed to scrape and with people that have given you their consent for their face's search.
