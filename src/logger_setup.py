import logging
import sys
import transformers
import datasets


def setup_logger(training_args) -> int:
    # Configure the root logger
    logging.basicConfig(
        level=training_args.get_process_log_level(),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    # Set verbosity for transformers and datasets
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger = logging.getLogger(__name__)
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info(f"Process log level: {logging.getLevelName(log_level)}")
    return log_level
