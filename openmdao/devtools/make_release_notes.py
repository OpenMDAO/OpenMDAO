"""

A command-line tool for creating the OpenMDAO release notes.

python -m openmdao.devtools.make_release_notes

Users will need to put their GitHub personal access token in the environment GITHUB_PAT to use this tool.

This will create a local cache of pull requests, saved in the path of this script.
If this cache does not exist, it will first pull all pull request from the last year.
This cache is saved as `{REPO_NAME}_pulls.json` (so is `OPENMDAO_pulls.json by default`).

Users can use this on other repositories as well with the -r or --repo option, which expects a string in owner/repo format.

Options
-------
  -h, --help       show this help message and exit
  -r, --repo REPO  GitHub repository in format owner/repo (default: OpenMDAO/OpenMDAO)

"""

from datetime import datetime, timezone, timedelta
import os
from pathlib import Path
from typing import Optional
import argparse

from github import Github, Auth
from pydantic import BaseModel, Field


# Use a PAT for private repos or to increase the rate limit of requests.
PERSONAL_ACCESS_TOKEN = os.environ['GITHUB_PAT']

# When creating a new cache, fetch PRs from this far back
INITIAL_LOOKBACK_DAYS = 365


class PullRequest(BaseModel):
    """Pydantic model for a GitHub pull request."""
    number: int
    title: str
    merged_at: datetime
    author: str = Field(default="unknown")
    url: str

    @classmethod
    def from_github_pr(cls, pr):
        """Create a PullRequest from a PyGithub PullRequest object."""
        return cls(
            number=pr.number,
            title=pr.title,
            merged_at=pr.merged_at.replace(tzinfo=timezone.utc) if pr.merged_at else None,
            author=pr.user.login if pr.user else "unknown",
            url=pr.html_url
        )


class PullRequestCache(BaseModel):
    """Container for cached pull requests."""
    pulls: list[PullRequest] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def get_most_recent_merge_date(self) -> Optional[datetime]:
        """Get the most recent merge date from cached pulls."""
        if not self.pulls:
            return None
        return max(pr.merged_at for pr in self.pulls)

    def add_pull(self, pr: PullRequest):
        """Add a pull request to the cache."""
        # Check if PR already exists (by number)
        if not any(p.number == pr.number for p in self.pulls):
            self.pulls.append(pr)

    def sort_by_merge_date(self):
        """Sort pulls by merge date (ascending)."""
        self.pulls.sort(key=lambda pr: pr.merged_at)

    def save(self, filepath: Path):
        """Save cache to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, filepath: Path) -> 'PullRequestCache':
        """Load cache from JSON file."""
        if not filepath.exists():
            return cls()

        with open(filepath, 'r') as f:
            return cls.model_validate_json(f.read())


def get_latest_release_date(repo) -> Optional[datetime]:
    """
    Get the date of the latest release from GitHub.

    Parameters
    ----------
    repo : github.Repository.Repository
        The GitHub repository object

    Returns
    -------
    Optional[datetime]
        The published date of the latest release, or None if no releases found
    """
    try:
        releases = repo.get_releases()
        latest_release = releases[0]
        release_date = latest_release.published_at.replace(tzinfo=timezone.utc)
        print(f"Latest release: {latest_release.tag_name} published on {release_date.date()}")
        return release_date
    except Exception as e:
        print(f"Warning: Could not get latest release: {e}")
        return None


def fetch_initial_pulls(repo, lookback_days=INITIAL_LOOKBACK_DAYS, state='closed', base='master'):
    """
    Fetch initial set of pull requests from the past year.

    Parameters
    ----------
    repo : github.Repository.Repository
        The GitHub repository object
    lookback_days : int
        Number of days to look back for initial fetch
    state : str
        One of either 'open' or 'closed'
    base : str
        The branch upon which the pull request is applied.

    Returns
    -------
    PullRequestCache
        A new cache populated with PRs from the lookback period
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    print(f"Creating new cache with PRs merged after {cutoff_date.isoformat()}")

    cache = PullRequestCache()
    pulls = repo.get_pulls(state=state, base=base, sort='updated', direction='desc')

    count = 0
    for pr in pulls:
        # Only process merged PRs
        if pr.merged_at is None:
            continue

        pr_merge_time = pr.merged_at.replace(tzinfo=timezone.utc)

        # Stop if we've gone past the lookback period
        if pr_merge_time < cutoff_date:
            break

        cache.add_pull(PullRequest.from_github_pr(pr))
        count += 1

        # Progress indicator for initial fetch
        if count % 50 == 0:
            print(f"  Fetched {count} PRs...")

    print(f"Initial cache created with {count} pull request(s)")
    cache.sort_by_merge_date()
    cache.last_updated = datetime.now(timezone.utc)

    return cache


def fetch_and_update_pulls(repo, cache: PullRequestCache, state='closed', base='master'):
    """
    Fetch pull requests from GitHub and update the cache.

    Only fetches PRs merged after the most recent cached PR to minimize API calls.

    Parameters
    ----------
    repo : github.Repository.Repository
        The GitHub repository object
    cache : PullRequestCache
        The existing cache to update
    state : str
        One of either 'open' or 'closed'
    base : str
        The branch upon which the pull request is applied.
    """
    most_recent = cache.get_most_recent_merge_date()

    if most_recent:
        print(f"Fetching PRs merged after {most_recent.isoformat()}")
    else:
        print("No cached PRs found. Fetching all merged PRs...")

    pulls = repo.get_pulls(state=state, base=base, sort='updated', direction='desc')

    new_count = 0
    for pr in pulls:
        # Only process merged PRs
        if pr.merged_at is None:
            continue

        pr_merge_time = pr.merged_at.replace(tzinfo=timezone.utc)

        # Stop if we've reached PRs we already have
        if most_recent and pr_merge_time <= most_recent:
            break

        cache.add_pull(PullRequest.from_github_pr(pr))
        new_count += 1

    if new_count > 0:
        print(f"Added {new_count} new pull request(s)")
        cache.sort_by_merge_date()
        cache.last_updated = datetime.now(timezone.utc)
    else:
        print("No new pull requests found")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch and cache GitHub pull requests, showing PRs since the latest release.'
    )
    parser.add_argument(
        '-r', '--repo',
        default='OpenMDAO/OpenMDAO',
        help='GitHub repository in format owner/repo (default: OpenMDAO/OpenMDAO)'
    )

    parser.add_argument(
        '-l', '--list-caches',
        action='store_true',
        help='Show the absolute path of any cache files.'
    )

    this_dir = Path(__file__).parent

    args = parser.parse_args()
    repo_name = args.repo

    if args.list_caches:
        cache_files = list(this_dir.glob("*_pulls.json"))
        if not cache_files:
            print('\nNo cache files found.')
        else:
            print('Existing cache files:')
            for f in cache_files:
                print(f'  {f}')
        exit(0)

    # Cache file location based on repo name
    cache_file = this_dir / f'{repo_name.split("/")[-1]}_pulls.json'

    print(f"Working with repository: {repo_name}")

    g = Github(auth=Auth.Token(PERSONAL_ACCESS_TOKEN))
    repo = g.get_repo(repo_name)

    # Check if cache exists
    if not cache_file.exists():
        print(f"Cache file not found. Creating initial cache from past {INITIAL_LOOKBACK_DAYS} days...\n")
        cache = fetch_initial_pulls(repo, lookback_days=INITIAL_LOOKBACK_DAYS)
    else:
        print(f"Loading existing cache from {cache_file}\n")
        cache = PullRequestCache.load(cache_file)
        # Fetch new PRs since last update
        fetch_and_update_pulls(repo, cache)

    # Save updated cache
    cache.save(cache_file)
    print(f"\nCache saved to {cache_file}")
    print(f"Total PRs in cache: {len(cache.pulls)}")
    print(f"Last updated: {cache.last_updated.isoformat()}")

    # Get latest release date
    print("\nChecking for latest release...")
    cutoff_date = get_latest_release_date(repo)

    # Display PRs since last release
    if cutoff_date:
        print("\nPull requests merged since last release:\n")
        matching_prs = [pr for pr in cache.pulls if pr.merged_at > cutoff_date]

        if matching_prs:
            for pr in matching_prs:
                print(f'- {pr.title} [#{pr.number}]({pr.url})')
        else:
            print("No pull requests found since the last release.")
    else:
        print("\nCould not determine latest release date. Showing all cached PRs:\n")
        for pr in cache.pulls:
            print(f'- {pr.title} [#{pr.number}]({pr.url})')


if __name__ == '__main__':
    main()
