from utils import plot_diffusion_schedules, create_all_diffusion_schedules

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, choices=["plot-schedules"], required=True)
    args = parser.parse_args()

    if args.command == "plot-schedules":

        ds = create_all_diffusion_schedules(T=1000)
        plot_diffusion_schedules(ds)
    
    else:
        raise ValueError(f"Unknown command {args.command}")