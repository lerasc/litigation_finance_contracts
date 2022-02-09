from setuptools import setup

setup(
    name            =  'LitFin_contracts',
    version         =  '1.0.0',
    packages        = ['LitFin_contracts'], 
    url             =  '',
    license         =  '',
    author          ='Sandro C. Lera',
    author_email    ='sandrolera@gmail.com',
    description     ='calculation of optimal litigation finance contracts',
    python_requires ='>3.5.2',
    install_requires=[
                        "numpy>=1.20.3",
                        "pandas>=1.3.4",
                        "seaborn>=0.11.2",
                        "matplotlib>=3.4.3",
                        "scipy>=1.7.2",
                        "hierarchical_grid_opt>=1.0.0",
                    ]
)
