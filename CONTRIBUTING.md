## Dev Install 
```
# Create and activate conda environment
conda create --name evalchemy python=3.10
conda activate evalchemy      

# Install devdependencies
pip install -e ".[dev]"
```
## Linting
```
# Use black before commiting
black .
```
