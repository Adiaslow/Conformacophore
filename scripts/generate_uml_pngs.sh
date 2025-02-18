pyreverse -o dot -p conformacophore src/conformacophore
dot -Grankdir=LR -Tpng classes_conformacophore.dot -o classes_conformacophore.png
pyreverse -o dot -p conformacophore src/conformacophore
dot -Grankdir=LR -Tpng packages_conformacophore.dot -o packages_conformacophore.png
