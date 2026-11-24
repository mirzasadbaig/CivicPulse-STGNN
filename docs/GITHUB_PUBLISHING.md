# Publishing checklist

The target is `mirzasadbaig/CivicPulse-STGNN`. Start private while checking ownership, internship/mentor approval, data rights and AI-disclosure requirements. The code license is deliberately not assigned. No repository creation or push is performed by this local package.

Create an empty private repository at https://github.com/new (do not initialize a README, license or gitignore). In the extracted project root:

```bash
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Initial CivicPulse-STGNN research release"
git remote add origin https://github.com/mirzasadbaig/CivicPulse-STGNN.git
git push -u origin main
```

Authenticate through GitHub's supported credential flow; never paste tokens in chat or commit them. Review the staged files before committing. Verify README, paper and evidence on GitHub. The HTML dashboard is runnable after cloning/downloading; it is not a hosted live dashboard. Keep private until approval, then change visibility if desired. A public URL is not evidence of peer review. For a private professor review, invite the professor as a collaborator or share the approved PDF separately.
