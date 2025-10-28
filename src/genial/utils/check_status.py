from genial.config.config_dir import ConfigDir
from loguru import logger


def main_cli():
    dir_config = ConfigDir(is_analysis=True)

    gener_out_dir_path = dir_config.generation_out_dir
    synth_out_dir_path = dir_config.synth_out_dir
    swact_out_dir_path = dir_config.swact_out_dir

    logger.info("Measuring length of gener_out ...")
    logger.info(f"gener_dir contains: {len(list(gener_out_dir_path.iterdir()))}")
    logger.info("Measuring length of synth_out ...")
    logger.info(f"synth_dir contains: {len(list(synth_out_dir_path.iterdir()))}")
    logger.info("Measuring length of swact_out ...")
    logger.info(f"swact_dir contains: {len(list(swact_out_dir_path.iterdir()))}")


if __name__ == "__main__":
    main_cli()
