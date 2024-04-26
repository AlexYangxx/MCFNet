# This is the offical code for MCFNet



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.
Pretrained model will come soon.

### Prerequisites

This project is based on Python, PyTorch.
* requirements
  ```sh
  ntorch>=0.4.1
  torchvision>=0.2.1
  dominate>=2.3.1
  visdom>=0.1.8.3
  ```

### Testing



1. Add the NIR images to the folder: /dataset/Validation, please make sure the png images contains "nir" in the filename. Otherwise, please do necssary rectification in /data/VCIP_nir2rgb_dataset.py,  class VCIPNir2RGBDataset_gen(BaseDataset): self.A_paths= [f for f in self.dir_A.glob('*nir*.png') if is_image(f)]
2. In launch.json, please find the "name": "Python: Generate_results", change the "--gpu_ids" to "0" or "1" accordingly.
3. Run the test-NIR.py
4. Go to "/final_results/results/test_latest_iter950/images", the generated RGB files are inside this folder. 


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Alex - yangxingxing817@gmail.com

Huiyu Zhai - wenyu.zhy@gmail.com


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Citation

Welcome to cite our paper if it inspires you!

   ```sh
     @misc{zhai2024multiscale,
      title={Multi-scale HSV Color Feature Embedding for High-fidelity NIR-to-RGB Spectrum Translation}, 
      author={Huiyu Zhai and Mo Chen and Xingxing Yang and Gusheng Kang},
      year={2024},
      eprint={2404.16685},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
      }
  ```

