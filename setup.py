import setuptools

setuptools.setup(
    name="DESmiles",
    version="0.1",
    author="D. E. Shaw Research",
    url="https://github.com/DEShawResearch/DESMILES",
    packages=["desmiles", "desmiles.decoding", "desmiles.scripts"],
    package_dir={"": "lib-python"},
    entry_points = {
        'console_scripts' :
            ['finetune-model=desmiles.scripts.finetune_model:main',
             'read-saved-model=desmiles.scripts.read_saved_model:main',
             'sample-variants-of-input=desmiles.scripts.sample_variants_of_input:main'
             ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7, <=3.9",
    install_requires=[
        "fastai>=1.0.55, <2",
        "seaborn>=0.10.1"
    ], # conda misreports rdkit as missing; "rdkit>=2020.03.3", 
)
