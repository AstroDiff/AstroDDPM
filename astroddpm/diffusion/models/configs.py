resunet_config = {
    "type": "resunet",
    "in_c": 1,
    "out_c": 1,
    "first_c": 10,
    "sizes" : [256, 128, 64, 32],
    "num_blocks" : 1,
    "n_steps" : 1000,
    "time_emb_dim" : 100,
    "dropout" : 0,
    "attention" : [],
    "normalisation" : "default",
    "padding_mode" : "circular",
    "eps_norm" : 1e-5,
    "skiprescale" : True,
    "discretization" : "discrete",
    "embedding_mode" : None,
    "has_phi" : False,
    "phi_embed_dim" : 100,
    "phi_shape" : None
}


ffresunet_config = {
    "type": "ffresunet",
    "in_c": 1,
    "out_c": 1,
    "first_c": 10,
    "sizes" : [256, 128, 64, 32],
    "num_blocks" : 1,
    "n_steps" : 1000,
    "time_emb_dim" : 100,
    "dropout" : 0,
    "attention" : [],
    "normalisation" : "default",
    "padding_mode" : "circular",
    "eps_norm" : 1e-5,
    "skiprescale" : True,
    "discretization" : "discrete",
    "embedding_mode" : None,
    "has_phi" : False,
    "phi_embed_dim" : 100,
    "phi_shape" : None,
    "n_ff_min" : 6,
    "n_ff_max" : 8,
}

def complete_config(config):
    if "type" not in config.keys():
        return complete_config({"type" : "resunet", **config})
    elif config["type"].lower() == "resunet":
        for key in resunet_config.keys():
            if key not in config.keys():
                config[key] = resunet_config[key]
        return config
    elif config["type"].lower() == "ffresunet":
        for key in ffresunet_config.keys():
            if key not in config.keys():
                config[key] = ffresunet_config[key]
        return config


# def get_network(config):
#     if "type" not in config.keys():
#         config["type"] = "ResUNet"
#     if config["type"].lower() == "resunet":
#         if "in_c" not in config.keys():
#             config["in_c"] = 1
#         if "out_c" not in config.keys():
#             config["out_c"] = 1
#         if "first_c" not in config.keys():
#             config["first_c"] = 10
#         if "sizes" not in config.keys():
#             config["sizes"] = [256, 128, 64, 32]
#         if "num_blocks" not in config.keys():
#             config["num_blocks"] = 1
#         if "n_steps" not in config.keys():
#             print("Warning: n_steps not specified, defaulting")
#             config["n_steps"] = 1000
#         if "time_emb_dim" not in config.keys():
#             config["time_emb_dim"] = 100
#         if "dropout" not in config.keys():
#             print("Warning: dropout not specified, defaulting")
#             config["dropout"] = 0
#         if "attention" not in config.keys():
#             print("Warning: attention not specified, defaulting")
#             config["attention"] = []
#         if "normalisation" not in config.keys():
#             print("Warning: normalisation not specified, defaulting")
#             config["normalisation"] = "default"
#         if "padding_mode" not in config.keys():
#             config["padding_mode"] = "circular"
#         if "eps_norm" not in config.keys():
#             config["eps_norm"] = 1e-5
#         if "skiprescale" not in config.keys():
#             config["skiprescale"] = True
#         if "discretization" not in config.keys():
#             config["discretization"] = "discrete"
#         if "embedding_mode" not in config.keys():
#             config["embedding_mode"] = None
#         if "has_phi" not in config.keys():
#             config["has_phi"] = False
#         if "phi_embed_dim" not in config.keys():
#             config["phi_embed_dim"] = 100
#         if "phi_shape" not in config.keys():
#             config["phi_shape"] = None
#         config.pop('type')
#         return ResUNet(**config)
#         #return ResUNet(in_c=config["in_c"], out_c=config["out_c"], first_c=config["first_c"], sizes=config["sizes"], num_blocks=config["num_blocks"], n_steps=config["n_steps"], time_emb_dim=config["time_emb_dim"], dropout=config["dropout"], attention=config["attention"], normalisation=config["normalisation"], padding_mode=config["padding_mode"], eps_norm=config["eps_norm"], skiprescale=config["skiprescale"], discretization=config["discretization"], embedding_mode=config["embedding_mode"], has_phi=config["has_phi"], phi_embed_dim=config["phi_embed_dim"], phi_shape=config["phi_shape"])
#     else:
#         raise NotImplementedError(f"Network type {config['type']} not implemented")