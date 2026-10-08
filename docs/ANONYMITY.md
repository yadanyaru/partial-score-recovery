# Anonymous review use

The export excludes personal paths, machine/user names, credentials, Git history,
internal review notes, manuscript revision scripts and development logs. Reference
JSON metadata uses relative resource paths. ZIP timestamps are normalized and its
entries have no owner names. Third-party public model/dataset authors and repository
identifiers remain because they identify the resources needed for reproduction.

Repository content and hosting identity are separate. Uploading these files to a
personal GitHub account can reveal the owner through the account, commit metadata,
issue history, profile, organization membership or repository links. For review,
use an anonymous repository/account with no identifying connections, or serve an
anonymized snapshot through a review-oriented mirror. Start from a fresh repository;
do not import the original project's Git history. Configure anonymous commit
metadata before committing and avoid identifying links in descriptions or issues.

The ARR call for papers requires linked software repositories to be properly
anonymized; its checklist also requires anonymized supplementary material:

- https://aclrollingreview.org/cfp
- https://github.com/acl-org/aclrollingreview/blob/main/authorchecklist.md

These are current ARR sources checked when preparing the export. The repository
does not assert that a future ACL 2027 submission policy is already final.

Run `python review.py verify` before uploading. Generated `runs/` and `cache/`
directories are ignored and should not be published; they may contain newly
generated local paths. No external repository has been created or uploaded by
this export process.

