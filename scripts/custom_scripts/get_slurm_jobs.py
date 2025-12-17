import subprocess

def get_slurm_jobs(user):
    cmd = [
        "squeue",
        "-u", user,
        "--Format=jobid,partition,name,state,nodelist",
        "--noheader"
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    jobs = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        jobid, partition, name, state, nodelist = line.split(maxsplit=6)
        jobs.append({
            "jobid": jobid,
            "partition": partition,
            "name": name,
            "state": state,
            "nodelist": nodelist,
        })

    return jobs

def should_cancel(job_name):
    return job_name != 'stay_alive'


def scancel_job(jobid):
    subprocess.run(
        ["scancel", jobid],
        check=True
    )

if __name__ == "__main__":
    jobs = get_slurm_jobs("ramaudruz")
    for job in jobs:
        if should_cancel(job["name"]):
            print(f"Cancelling job {job['jobid']} ({job['name']})")
            scancel_job(job["jobid"])