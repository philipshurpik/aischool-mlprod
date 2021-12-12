# serving face_parser with different model servers

### Install face_parser as package
```
python setup.py install
```

### Serve BentoML
```
python make_service.py
bentoml serve FaceParserService:latest
```