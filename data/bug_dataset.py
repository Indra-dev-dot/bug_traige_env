import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import BugReport, Severity, Team

BUGS_WITH_LABELS = [
    {
        "bug": BugReport(
            id="BUG-001",
            title="Production database returning 500 errors for all write operations",
            description=(
                "All write operations to the production PostgreSQL database are failing "
                "with 500 Internal Server Error. Started at 14:23 UTC. Reads still work. "
                "Error logs show: FATAL: remaining connection slots are reserved. "
                "Affects all users — cannot place orders or save any data."
            ),
            steps_to_reproduce=[
                "Go to any page that writes data",
                "Perform any write action",
                "Observe 500 error",
            ],
            environment="Production — all browsers, all platforms",
            reporter="on-call-engineer",
            created_at="2024-03-15T14:25:00Z",
        ),
        "ground_truth": {
            "severity": Severity.P0,
            "team": Team.INFRA,
            "root_cause_hint": "database connection pool exhausted",
        },
    },
    {
        "bug": BugReport(
            id="BUG-002",
            title="Authentication tokens not invalidated on logout — security issue",
            description=(
                "After a user logs out, their JWT token remains valid for the full "
                "7-day expiry window. An attacker who intercepts a token can continue "
                "using it even after the user has logged out. Confirmed by capturing "
                "token via browser devtools, logging out, then making authenticated "
                "API calls — all requests succeed."
            ),
            steps_to_reproduce=[
                "Log in and capture JWT from localStorage",
                "Log out via the logout button",
                "Make API call with captured JWT",
                "Observe: request succeeds despite being logged out",
            ],
            environment="All environments — API v2",
            reporter="security-researcher",
            created_at="2024-03-14T09:10:00Z",
        ),
        "ground_truth": {
            "severity": Severity.P0,
            "team": Team.SECURITY,
            "root_cause_hint": "token blacklist not implemented on logout",
        },
    },
    {
        "bug": BugReport(
            id="BUG-003",
            title="Payment checkout breaks when discount code is applied",
            description=(
                "When a user applies a valid discount code during checkout, the order "
                "total updates correctly but clicking Place Order does nothing — "
                "no error, no confirmation, just a spinner that runs forever. "
                "Removing the discount code and retrying works fine."
            ),
            steps_to_reproduce=[
                "Add any item to cart",
                "Go to checkout",
                "Enter discount code SAVE10",
                "Click Apply — price updates correctly",
                "Click Place Order",
                "Observe: infinite spinner, no order placed",
            ],
            environment="Chrome 120, Firefox 121 — macOS and Windows",
            reporter="qa-team",
            created_at="2024-03-13T11:30:00Z",
        ),
        "ground_truth": {
            "severity": Severity.P1,
            "team": Team.BACKEND,
            "root_cause_hint": "discount code causes silent validation failure",
        },
    },
    {
        "bug": BugReport(
            id="BUG-004",
            title="Mobile app crashes on iOS 17 when opening notification settings",
            description=(
                "The iOS app hard crashes whenever the user navigates to "
                "Settings → Notifications. Introduced in v3.2.1 last week. "
                "Affects all iPhone models running iOS 17+. Users on iOS 16 "
                "are unaffected. Crash report: EXC_BAD_ACCESS in NotificationSettingsView."
            ),
            steps_to_reproduce=[
                "Open the app on iOS 17 device",
                "Tap profile icon then Settings",
                "Tap Notifications",
                "App crashes immediately",
            ],
            environment="iOS 17.0-17.3, all iPhone models, app v3.2.1",
            reporter="user-complaint-bot",
            created_at="2024-03-12T16:45:00Z",
        ),
        "ground_truth": {
            "severity": Severity.P1,
            "team": Team.MOBILE,
            "root_cause_hint": "iOS 17 API change in notification permissions handling",
        },
    },
    {
        "bug": BugReport(
            id="BUG-005",
            title="Search results don't update when filters changed without re-clicking search",
            description=(
                "When a user applies a filter after performing a search, the results "
                "don't update automatically. They have to click the Search button again. "
                "Workaround: click Search after applying any filter."
            ),
            steps_to_reproduce=[
                "Search for laptop",
                "Change the price filter to 500-1000",
                "Observe: results don't update",
                "Click Search again — now results update correctly",
            ],
            environment="All browsers, web app",
            reporter="product-team",
            created_at="2024-03-11T10:00:00Z",
        ),
        "ground_truth": {
            "severity": Severity.P2,
            "team": Team.FRONTEND,
            "root_cause_hint": "filter onChange handler not triggering search re-fetch",
        },
    },
    {
        "bug": BugReport(
            id="BUG-006",
            title="CSV export contains duplicate rows when more than 1000 records",
            description=(
                "When exporting data to CSV with more than 1000 records, the exported "
                "file contains duplicate rows — the first page of results appears twice. "
                "Files with fewer than 1000 records export correctly."
            ),
            steps_to_reproduce=[
                "Navigate to Reports then Export",
                "Select a dataset with 1000+ records",
                "Click Export as CSV",
                "Open the file — first 100 rows appear twice",
            ],
            environment="Web app — all browsers",
            reporter="data-analyst-user",
            created_at="2024-03-10T14:20:00Z",
        ),
        "ground_truth": {
            "severity": Severity.P2,
            "team": Team.BACKEND,
            "root_cause_hint": "pagination offset bug in export query — page 1 fetched twice",
        },
    },
    {
        "bug": BugReport(
            id="BUG-007",
            title="Typo in error message: Plaese try again instead of Please try again",
            description=(
                "Minor typo in the generic error message displayed when a form "
                "submission fails. The message reads Plaese try again later instead "
                "of Please try again later. Seen on the contact form and feedback form."
            ),
            steps_to_reproduce=[
                "Go to Contact Us page",
                "Submit the form with an invalid email",
                "Observe typo in error message",
            ],
            environment="All browsers, web app",
            reporter="content-team",
            created_at="2024-03-09T09:00:00Z",
        ),
        "ground_truth": {
            "severity": Severity.P3,
            "team": Team.FRONTEND,
            "root_cause_hint": "hardcoded string with typo in error handler",
        },
    },
    {
        "bug": BugReport(
            id="BUG-008",
            title="Dashboard loading spinner stuck after data loads on slow connections",
            description=(
                "On slow 3G connections, the loading spinner on the dashboard continues "
                "spinning even after the data has loaded. Refreshing the page fixes it. "
                "On normal connections the spinner stops correctly."
            ),
            steps_to_reproduce=[
                "Open Chrome DevTools, Network, set throttling to Slow 3G",
                "Navigate to Dashboard",
                "Wait for data to load",
                "Observe: spinner still showing even though data is visible",
            ],
            environment="Chrome with Slow 3G throttling",
            reporter="frontend-qa",
            created_at="2024-03-08T15:30:00Z",
        ),
        "ground_truth": {
            "severity": Severity.P3,
            "team": Team.FRONTEND,
            "root_cause_hint": "loading state not reset when fetch completes on slow connection",
        },
    },
]

DUPLICATE_BUGS = [
    {
        "new_bug": BugReport(
            id="BUG-009",
            title="Can't log out properly — session stays active",
            description=(
                "I logged out of my account but when I went back to the site I was "
                "still logged in. My session seems to persist even after clicking logout. "
                "Tested in incognito — still able to access my account with the old token."
            ),
            steps_to_reproduce=[
                "Log in",
                "Save auth token from browser storage",
                "Click logout",
                "Use saved token to make API request — it works",
            ],
            environment="Chrome 119, Windows 11",
            reporter="end-user-42",
            created_at="2024-03-15T16:00:00Z",
        ),
        "is_duplicate_of": "BUG-002",
    },
    {
        "new_bug": BugReport(
            id="BUG-010",
            title="Dark mode toggle not saving preference",
            description=(
                "When I switch to dark mode and refresh the page, the setting resets "
                "to light mode. My dark mode preference is not being saved. "
                "This happens on all browsers."
            ),
            steps_to_reproduce=[
                "Go to Settings then Appearance",
                "Toggle Dark Mode on",
                "Refresh the page",
                "Observe: back to light mode",
            ],
            environment="All browsers",
            reporter="end-user-99",
            created_at="2024-03-15T17:00:00Z",
        ),
        "is_duplicate_of": "none",
    },
]