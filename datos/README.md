# Datos

Los datasets no se suben al repositorio por su tamaño (~1 GB en total). Descárgalos y déjalos con esta estructura:

```
datos/
├── cifar-10-batches-py/              data_batch_1..5, test_batch, batches.meta
├── oxford-iiit-pet/
│   ├── images/
│   └── annotations/
└── caltech101/
    └── 101_ObjectCategories/         una carpeta por clase
```

| Dataset | Descarga |
|---|---|
| CIFAR-10 (versión Python) | https://www.cs.toronto.edu/~kriz/cifar.html |
| Oxford-IIIT Pet | https://www.robots.ox.ac.uk/~vgg/data/pets/ |
| Caltech-101 | https://data.caltech.edu/records/20086 |

En Caltech-101 se excluye la clase `BACKGROUND_Google`, y los notebooks lo hacen solos.
