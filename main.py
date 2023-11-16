import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, choices=["plot-schedules", "plot-sinusoidal"], required=True)
    args = parser.parse_args()

    if args.command == "plot-schedules":
        from utils import plot_diffusion_schedules, create_all_diffusion_schedules

        ds = create_all_diffusion_schedules(T=1000)
        plot_diffusion_schedules(ds)
    
    elif args.command == "plot-sinusoidal":
        from utils import plot_sinusoidal_embeddings

        plot_sinusoidal_embeddings(noise_embedding_size=32)

    else:
        raise ValueError(f"Unknown command {args.command}")