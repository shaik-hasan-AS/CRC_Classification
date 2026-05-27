import wandb
api = wandb.Api()
run = api.run("leogasia7-vellore-institute-of-technology/MedLite-CRC/o6ie2qg3")
history = run.history()
print("COLUMNS:")
print(history.columns.tolist())
