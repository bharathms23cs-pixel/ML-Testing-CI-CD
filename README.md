Project structure

ML-Testing-CI-CD/
│
├── .github/
│   └── workflows/
│       ├── ml-test.yml          # 1. Your ML testing CI/CD
│       └── python-publish.yml   # 2. For PyPI publishing
│
├── tests/
│   └── test_model.py            # Tests your model.pkl
│
├── model.pkl                    # Trained model (rename from model (1).pkl)
├── house.csv                    # Dataset (rename from house (2).csv)
├── requirements.txt             # All packages
├── README.md                    # Installation link
└── .gitignore

