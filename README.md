# M4Human

## 1. Environment Setup
- Create the environment using:
  ```sh
  conda env create -f environment.yml
  ```
- Our point-based method require CUDA pointnet++ acceleration, follow the setup instructions in the P4Transformer repository: [https://github.com/erikwijmans/Pointnet2_PyTorch].

## 2. Download Datasets
- **Full raw dataset:** [URL] Comming Soon
- **Full processed dataset:** [URL] Comming Soon
- **Full processed dataset (radar modality):** [URL] Comming Soon
- **Sample Vis dataset:** [https://entuedu-my.sharepoint.com/:f:/g/personal/fanj0019_e_ntu_edu_sg/IgDSzyJNtxJYRZn3lXtWjaM7AYSyj3p1IH0h2ziGGREplO0?e=d1Y85Y]

**After downloading, organize your folders as follows (we recommend keeping datasets outside the repo):**

```
MR-Mesh-main1/                      # Main repo folder
mmDataset/                          # (Full processed dataset)
    MR-Mesh/
        rf3dpose_all/
            calib.lmdb
            image.lmdb
            depth.lmdb
            radar_pc.lmdb           # (RPC)
            params.lmdb             # (GT params)
            indeces.pkl.gz          # (dataset split configuration)
            ... (other .lmdb and .lock files)
cached_data_test_vis/               # (Sample Vis dataset)
    rf3dpose_all/
        calib.lmdb
        image.lmdb
        depth.lmdb
        radar_pc.lmdb         
        params.lmdb           
        indeces.pkl.gz        
        ... (other .lmdb and .lock files)

```

## 3. Download SMPL Models
- Download SMPL models from official source or URL.
- Place them in models/ with the following structure:
  ```
    MR-Mesh-main1/
        models/
            smplx/
            SMPLX_FEMALE.npz
            SMPLX_FEMALE.pkl
            SMPLX_MALE.npz
            SMPLX_MALE.pkl
            SMPLX_NEUTRAL.npz
            SMPLX_NEUTRAL.pkl
            smplx_npz.zip
            version.txt
  ```

## 4. Demo & Visualization
- We provide `demo.ipynb` for:
  - Dataset vis demo example: generate modality GIFs and save to `vis_depth/`
  - Preprocessed radar dataloader tutorial

---
For more details, see comments in the code and notebook.


## 5. Benchmarking (Radar Modality Only)
- To run the benchmark:
  ```sh
  torchrun --nproc_per_node=4 main1_multigpu.py
  ```
- Select model/config by editing `main1_multigpu.py`.

## 6. RGBD Modality
- RGBD support will be released after further organization.


