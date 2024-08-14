# RQ4: Practicality Analysis

How effective is PatchFinder in real-world applications, particularly when detecting security patches for CVEs without associated trace links?

Upon deploying PatchFinder on this curated set, we derived the top-10 ranked outputs for each CVE. From this pool, we initially
manually reviewed and traced 533 patches. These patches achieved a ranking of 1.65 in PatchFinder ’s output on average. The entire review process was efficiently conducted, taking a total of 13.31 man-hours. We then submitted these patches to CNAs for review. Notably, 482 of these were subsequently confirmed by CNAs so far.

## Traced Patches

All patches that were traced by PatchFinder were manually reviewed by the authors. The patches were then submitted to CNAs for review. The patches that were submitted and confirmed by CNAs are available in [./PatchFinder-ISSTA2024_RQ4.csv](./PatchFinder-ISSTA2024_RQ4.csv).

## News

- NEW！🎉 As of June 1st, 2024, 612 ones (out of the 700 submitted) have been confirmed by CNA! 

- NEW！🎉 As of May 20th, 2024, we have extended our experiments and submitted 700 patch commits PatchFinder found to the official CNAs, of which 600 received official acknowledgment (details in the Google Sheet below)! The others are pending confirmation. 
