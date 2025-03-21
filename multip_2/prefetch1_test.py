import multiprocessing as mp

ctx = mp.get_context("spawn")
(child, pipe) = ctx.Pipe(duplex=True)
job_process = ctx.Process(
    name="my_job",
    target=job_func,
    args=(
        child,
        server_info,
        manager,
        job_config,
        config_file,
    ),
)
job_process.start()