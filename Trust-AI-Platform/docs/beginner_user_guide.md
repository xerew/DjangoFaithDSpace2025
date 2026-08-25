<div class="cover-page">
  <div class="cover-mark">BEGINNER USER GUIDE</div>
  <img class="cover-logo" src="../static/img/logo.png" alt="Trust AI Lab logo">
  <h1>Trust AI Lab</h1>
  <p class="cover-subtitle">Create, deliver, and improve inquiry-based learning scenarios</p>
  <div class="cover-flow">
    <span>AUTHOR</span><b>→</b><span>DELIVER</span><b>→</b><span>ANALYSE</span><b>→</b><span>IMPROVE</span>
  </div>
  <div class="cover-meta">
    <strong>Audience</strong> Learners, teachers, organization administrators, and platform administrators<br>
    <strong>Edition</strong> July 2026 &nbsp;·&nbsp; English
  </div>
</div>

<div class="page-break"></div>

# Welcome

Trust AI Lab is a learning platform for building and running inquiry-based scenarios. A teacher can combine explanatory content, questions, simulations, remote laboratories, and VR/AR activities into a route through a lesson. Learner activity is then summarised in dashboards, risk indicators, and optional AI-assisted recommendations.

This guide is written for a first-time user. Begin with the **Quick start** for your role, then return to the detailed sections when you need them.

> **Important:** The menus you see depend on your account role. If an item described here is missing, it may not be enabled for your role or organization. Ask your local platform administrator before assuming something is broken.

## The platform in one picture

<div class="process-grid">
  <div><b>1. Design</b><span>Scenario → phases → activities</span></div>
  <div><b>2. Connect</b><span>Answers and performance routes</span></div>
  <div><b>3. Deliver</b><span>Student View and student groups</span></div>
  <div><b>4. Learn</b><span>Dashboards and activity risk</span></div>
  <div><b>5. Improve</b><span>Review AI proposals and create a copy</span></div>
</div>

## Contents

[TOC]

<div class="page-break"></div>

# 1. Roles and access

| Role | Typical access |
|---|---|
| **Learner / student** | Browse assigned or visible scenarios, complete activities and pre-scenario questions, submit feedback. |
| **Teacher** | Everything needed to create scenarios, create student groups, run scenarios, view analytics, review AI proposals, message colleagues, and join organizations. |
| **Organization administrator** | Manage an organization's details, members, join requests, announcements, and organization chat. |
| **Platform administrator** | Manage users and roles, the experiment/lab catalogue, platform health, and AI recommendation controls. |

Your account can have more than one responsibility. For example, a teacher may also administer an organization.

## Sign in

1. Open the platform address supplied by your school or administrator.
2. Enter your **username** and **password**, then select **Sign In**.
3. If you have forgotten your password, choose the password-reset link and follow the email instructions.
4. If a generated student account has never signed in before, use the initial password supplied by the teacher.

> **Security tip:** Do not share an account. Change an initial password after the first sign-in, and sign out on shared computers.

## Main navigation

Use the menu button at the top left to collapse or open the sidebar. Teachers normally see:

| Menu | What it is for |
|---|---|
| **Home** | Welcome page, recent public scenarios, and shortcuts. |
| **Dashboard** | Scenario analytics, group/date filters, charts, and CSV export. |
| **Authoring Tool** | Find, create, import, edit, duplicate, export, and delete scenarios. |
| **Student Groups** | Generate student accounts and assign scenarios. |
| **Student View** | Experience a scenario as a learner would. |
| **Documentation & Tutorials** | Guides and support resources. |

The profile menu at the top right contains **Messages**, **Account Settings**, **Organizations**, administrator tools when permitted, and **Sign Out**.

<div class="page-break"></div>

# 2. Quick start

## Teacher: publish a small scenario

For your first attempt, build a short private scenario with one phase, one content activity, and one question.

1. Go to **Authoring Tool** and select **Create Scenario**.
2. Choose **Manual Setup**.
3. Enter a unique name, description, learning goals, student age range, subjects, language, and suggested learning time.
4. Save the scenario. New scenarios are private by default.
5. Open it and select **Add Phase**. Give the phase a meaningful name such as “Explore”.
6. Open the phase and select **Create Activity**.
7. Create a short content activity and choose its **Next Activity** as **Create New Activity**.
8. Edit the new activity, change its type to **Question**, and write the question in **Content**.
9. Open the question, choose **Create Answers**, add the answer options, set weights, and mark exactly one answer as correct.
10. Open **Student View**, find your scenario, and complete it from start to finish.
11. Return to **Authoring Tool → Edit Scenario** and choose the visibility you need.

> **Beginner rule:** Test while the scenario is still **Private**. Make it organization-visible or public only after every route has been tried.

## Teacher: run it with a class

1. Open **Student Groups → Create New Group**.
2. Enter a group name, a username prefix, and the number of student accounts.
3. Assign the scenario to the group and save.
4. Download the credentials and distribute each username/password privately.
5. Ask learners to sign in, open **Student View**, and start the assigned scenario.
6. After the session, open **Dashboard**, select the scenario and group, and choose **Apply**.

## Learner: complete a scenario

1. Sign in with the credentials supplied by your teacher.
2. Open **Student View**.
3. Search or filter for the assigned scenario, then select **Start**.
4. If pre-scenario questions appear, answer them and confirm submission.
5. Select **I am ready** and follow each activity.
6. Read any feedback before moving on. Some answers lead to different next activities.
7. Use **Restart** only when you intend to begin the scenario again.

<div class="page-break"></div>

# 3. Understand scenario structure

A scenario is not simply a list of pages. It is a connected learning route.

<div class="hierarchy">
  <div class="h-level h-one"><b>SCENARIO</b><span>Purpose, audience, visibility, and entry activity</span></div>
  <div class="h-arrow">↓</div>
  <div class="h-level h-two"><b>PHASES</b><span>Major stages of the inquiry process</span></div>
  <div class="h-arrow">↓</div>
  <div class="h-level h-three"><b>ACTIVITIES</b><span>Content, questions, experiments, and evaluations</span></div>
  <div class="h-arrow">↓</div>
  <div class="h-level h-four"><b>ROUTES</b><span>Next activity selected directly, by answer, or by performance</span></div>
</div>

## Scenario

The scenario holds the lesson-wide information: name, description, learning goals, student age range, subjects, language, suggested time, cover image, visibility, and organizations.

The first activity created is automatically used as the scenario's starting point unless an administrator configures another entry activity.

## Phase

A phase groups related activities. Use phases to express the teaching design rather than to create arbitrary folders. A common inquiry sequence is:

1. Orientation or engagement
2. Conceptualisation
3. Investigation or exploration
4. Conclusion
5. Discussion or reflection

The interface permits up to five phases per scenario.

## Activity

An activity is one learner-facing step. Important activity options include:

- **Name** — short and identifiable in routes and reports.
- **Content** — the instructions, explanation, question, images, and other rich text shown to learners.
- **Activity Type** — determines behaviour, such as a Question or Experiment.
- **Helping Quote** — optional guidance for learners.
- **Evaluatable Activity** — calculates a performance result from selected activities.
- **Next Activity** — the direct route for a normal, non-question activity.

For an **Experiment**, choose the relevant subtype and catalogue entry: Simulation, Remote Lab, or VR/AR Lab. Preview the selected resource before saving.

## Answers and routes

Question activities need answer options. Each answer has:

- answer text;
- a weight from 1 to 3;
- a correct/incorrect setting; and
- for non-evaluatable questions, an optional next activity.

Exactly one answer must be marked correct in the answer editor. Different answers can point to different next activities, which creates adaptive branches.

> **Route rule:** A question without answers is incomplete. An answer without a next route may intentionally finish a branch, but check that this is really what you want.

<div class="page-break"></div>

# 4. Create and edit a scenario

## Find an existing scenario

Open **Authoring Tool**. You can search by name, description, or creator and filter by date, language, subject, visibility, or **Show My Scenarios**.

Each scenario card offers actions according to your permissions:

- **View** — open its details and phases.
- **Duplicate** — create an editable copy.
- **Edit** — change scenario-level information and visibility.
- **Delete** — permanently remove the scenario; confirm carefully.

## Create manually

Select **Create Scenario → Manual Setup** and complete the required fields. Prefer a unique, descriptive name such as “Pendulum Investigation — Ages 13–15” rather than “Physics Lesson”.

After saving:

1. Add the phases.
2. Open each phase and add its activities.
3. Connect normal activities with **Next Activity**.
4. Add answers and answer routes to questions.
5. Configure evaluation criteria where used.
6. Preview in **Student View**.

## Import from a template

Select **Create Scenario → Upload Template** for a faster, spreadsheet-based setup.

- Download the blank `.xlsx` template, fill it in without renaming or removing required sheets/columns, then upload it.
- To include images, upload a `.zip` containing the spreadsheet at the ZIP root and images inside an `images/` folder.
- An optional scenario cover image can be uploaded separately.
- If validation errors appear, correct every listed sheet/row/column issue and upload again.

An exported platform ZIP already has the expected structure and is the safest starting point for a complex scenario.

## Export and backup

Open a scenario and select **Export Scenario**. The downloaded ZIP contains an Excel representation and its images. Export before major restructuring so you have a restorable copy.

## Add RAG documents

Scenario owners can add PDF reference documents in the **RAG Documents** area. These documents provide scenario-specific context to AI generation. Only upload material you are authorised to use and review or delete outdated documents before regenerating AI context.

<div class="page-break"></div>

# 5. Build reliable learning routes

## Direct route

Use **Next Activity** on a normal content or experiment activity.

<div class="route-diagram"><span>Introduction</span><b>→</b><span>Simulation</span><b>→</b><span>Reflection</span></div>

## Answer-based route

Use the **Next Activity** selector beside each question answer.

<div class="branch-diagram">
  <div class="branch-root">Question</div>
  <div class="branch-lines">
    <div><b>Correct</b><span>→ Extension activity</span></div>
    <div><b>Incorrect</b><span>→ Hint and retry activity</span></div>
  </div>
</div>

## Performance-based route

Turn on **Evaluatable Activity**, choose the activities to evaluate, and optionally mark it as the phase's **Primary Evaluation**. In **Create Evaluation Criterion**, choose the next activity for High, Moderate, and Low performers.

The route uses average answer weight from 1 to 3:

- **High** — average weight at or above the High threshold (default recommendation: 2.5).
- **Moderate** — average weight at or above the Moderate threshold (default recommendation: 1.5).
- **Low** — everyone below the Moderate threshold.

## Route quality checklist

- [ ] The first activity introduces the task and does not assume earlier context.
- [ ] Every Question activity contains answers.
- [ ] Exactly one answer is marked correct where required.
- [ ] Every intended branch has a next activity.
- [ ] Remediation branches return to the main flow or end deliberately.
- [ ] Evaluation thresholds are ordered High > Moderate > Low.
- [ ] There are no accidental loops.
- [ ] Every activity can be reached from the start.
- [ ] Every end point gives learners a clear completion message.

> **Safe editing habit:** After deleting or moving an activity, retest every route that previously entered or left it.

<div class="page-break"></div>

# 6. Visibility, sharing, and organizations

Edit a scenario to choose its visibility:

| Visibility | Who can see it |
|---|---|
| **Private — only you** | Best while authoring and testing. |
| **Organization — selected organizations** | Members of selected organizations. You may also allow organization members to edit it. |
| **Public — everyone** | All eligible platform users. Personalised AI-created copies cannot be made public. |

When using organization visibility, select the organizations explicitly. Enabling **Allow organization members to edit this scenario** gives collaborators significant control, so use it only with a trusted authoring group.

## Organizations

Open the profile menu and choose **Organizations**. Depending on permissions, users can:

- browse organizations and request to join;
- view members and announcements;
- use organization chat;
- create or edit an organization;
- add, promote, demote, or remove members; and
- approve or reject join requests.

Organization administrators should post important changes as announcements and use chat for discussion. Do not place student passwords or sensitive learner data in announcements or chat.

## Messages

Teachers can open **Messages** from the profile menu. Start a conversation from a colleague's profile or an organization's member list. A badge and notification toast indicate unread messages.

<div class="page-break"></div>

# 7. Student groups and classroom delivery

## Create a group

1. Go to **Student Groups** and select **Create New Group**.
2. Enter a recognisable group name.
3. Enter a **User Prefix**. The platform uses it when generating usernames.
4. Enter the number of students.
5. Select one or more scenarios to assign.
6. Save the group.

The group page shows generated usernames and initial passwords. Select **Download Credentials** to save a copy.

## Handle credentials safely

- Give each learner only their own credentials.
- Avoid projecting the full credentials table in class.
- Store downloaded credential files securely and delete local copies when no longer needed.
- Ask students to change initial passwords where your deployment allows it.
- Never include passwords in analytics exports or public documents.

## Edit a group

Use **Edit Group** to change its name, number of students, or assigned scenarios. Review credential changes after increasing the group size. Deleting a group is a destructive action; download any records you need first.

## Before the session

1. Complete the scenario yourself in Student View.
2. Test embedded simulations/labs on the classroom network and devices.
3. Verify the scenario is assigned to the correct group.
4. Check that student accounts can sign in.
5. Decide whether learners may restart.
6. Keep a short offline activity ready in case an external lab is unavailable.

<div class="page-break"></div>

# 8. Learner guide

## Find your scenario

Open **Student View**. Search by name or filter by language and subject. Only scenarios visible to you or assigned through your group should appear.

## Start and progress

1. Select **Start** on the scenario card.
2. Complete any pre-scenario form and confirm before submission.
3. On the introduction screen, select **I am ready**.
4. Read the activity name and content fully.
5. For a question, choose the answer that best fits and submit it.
6. For an experiment, follow its instructions and wait for the resource to load.
7. Read feedback and continue to the next activity.

Your route may differ from another learner's because answers and performance can lead to different activities. This is expected.

## Restart or leave

**Restart** begins the scenario again. Use it only when instructed, because a restart can affect the attempt history. To work on another scenario, return to the scenario list.

## Submit feedback

An active feedback form may appear before, during, or after a scenario. Required questions must be completed. Feedback helps improve the learning experience; do not include passwords or unnecessary personal information.

## If something goes wrong

- Refresh once if an activity remains blank.
- Allow required pop-ups, camera, microphone, or VR permissions only for a trusted lab activity.
- If a simulation or remote lab fails, note the scenario and activity names and tell your teacher.
- Do not repeatedly submit the same answer while the page shows a loading state.

<div class="page-break"></div>

# 9. Dashboard and analytics

The teacher **Dashboard** summarises learner activity. First select a scenario. Optionally select student groups and a date range, then choose **Apply**.

Common reports include:

- student flow through the scenario;
- activity answer distributions;
- performance overview;
- average time per activity or phase;
- detailed phase scores;
- numbers of Low, Moderate, and High performers;
- time by performer type; and
- most common scenario and phase paths.

Charts may take a few seconds to generate. Keep the page open while the loading message is visible.

## Export learner metrics

Choose **Generate & Download CSV** after selecting a scenario. You can include per-activity scores and timing. Treat the CSV as sensitive educational data: store it securely, share it only with authorised people, and follow your institution's retention policy.

## Interpret responsibly

Analytics show patterns, not motives. A long time on an activity may mean productive investigation, unclear instructions, a technical problem, or distraction. Combine chart evidence with classroom observation and learner feedback before changing instruction.

<div class="page-break"></div>

# 10. Metrics, risk, and AI proposals

Open a scenario and choose **Metrics & AI**. Depending on your permission, the page provides up to four related actions:

Any teacher who can access a public or organization-visible scenario can open its Metrics & AI page, view its metrics, review the current shared proposals, and create a personalised scenario from their own decisions. Proposal decisions and edits are stored separately for each teacher. Only the original scenario creator and platform administrators can use **Generate LLM Context** or **Regenerate All Context** to create a new proposal set.

1. **Performance Metrics** — calculate category-level performance.
2. **Activities in Risk** — identify activities with concerning patterns.
3. **Generate LLM Context** — analyse activities not processed previously and generate proposals.
4. **Regenerate All Context** — intentionally rerun analysis for all activities.

The platform warns when the scenario has fewer than its configured minimum implementations (200 by default). Low-data metrics and proposals may be unstable, so treat them as exploratory.

## Understand proposal actions

| Action | Meaning |
|---|---|
| **Create** | Insert a new supporting activity after an existing activity. A proposal is not inserted before the scenario's start activity. |
| **Revise** | Change the content or structure of an existing activity. |
| **Skip** | Recommend no content change for that risk signal. |

The proposal card shows the risk category, activity, proposed content, answer options where relevant, explanation, and an insertion preview for Create proposals.

## Review proposals

For every proposal:

1. Read the activity, proposal type, explanation, content, and routes.
2. For a Question proposal, verify that answer choices are present, clearly lettered (for example, **A. Answer text**), and that exactly one is correct.
3. Select **Edit** to correct the name, content, explanation, or answers before deciding.
4. Select **Accept** only when the proposal is pedagogically appropriate and structurally complete.
5. Select **Reject** and choose the closest reason when it should not be used.

Proposal states are personal to the reviewer:

- **Pending** — no final decision.
- **Accepted** — will be applied to the personalised copy.
- **Rejected** — will not be applied.
- **Undo Accept** or **Undo Reject** — returns the proposal to **Pending**; it does not switch directly to the opposite decision.

When all proposals have been reviewed, select **Create Personalised Scenario**. The platform creates a private copy and applies your accepted proposals. The original scenario remains unchanged. Creation runs in the background, so allow a short delay before looking for the copy in Authoring Tool.

> **Human-review rule:** AI output is a draft, never an automatic teaching decision. Check factual accuracy, age appropriateness, accessibility, answer correctness, routes, and source permissions.

## Proposal history

Use **History** to inspect previous generation runs and see how proposals were decided at that time. Regenerating proposals starts a new current set; history helps you compare runs without mixing them.

<div class="page-break"></div>

# 11. Feedback, profile, and account settings

## Feedback forms

Authorised users can create feedback forms, choose the audience, add required or optional questions, assign forms to all or selected scenarios, and activate them. Responses can be reviewed and exported as CSV or XLSX.

Use concise questions and collect only information needed for a clear teaching or evaluation purpose.

## Update your profile

Open the profile menu and choose **Account Settings**. The profile area can contain your name, email, username, country, gender, institution, profile picture, and short biography. Save changes before leaving the tab.

## Change your password

Open the **Change Password** tab, enter the current password, then enter and confirm a new password. Use a long, unique password that is not used on another service.

## Sign out

Always use **Sign Out** from the profile menu when using a shared or classroom computer. Closing a browser tab may leave the session active.

<div class="page-break"></div>

# 12. Administrator essentials

This section is a short orientation, not a replacement for your institution's governance procedures.

## Platform administration

Authorised administrators can manage:

- user accounts, roles, activation state, and safe impersonation for support;
- simulations, remote labs, and VR/AR lab catalogue entries;
- scenario start activities and scenario graph health;
- structural AI-generation failures; and
- recommendation policy and bandit statistics, including action/context reward counts and algorithm controls.

When impersonating a user, the orange banner indicates that you are viewing the platform as that person. Select **Exit Impersonation** immediately after support work.

## Scenario health

Before applying AI proposals or publishing a complex scenario, check for:

- a missing or invalid start activity;
- unreachable activities;
- questions with no answers;
- missing answer routes;
- routes pointing outside the scenario;
- unintended cycles; and
- branches that terminate unexpectedly.

Structural generation failures should be recorded separately from pedagogical rejection. A malformed AI response is a system-quality issue, not evidence that the teacher disliked a valid teaching suggestion.

## AI controls

Recommendation statistics should be monitored by action and context. Early exploration policy and mature UCB/Thompson behaviour affect which actions are sampled, but teachers must always retain the final decision. Change these controls only with an evaluation plan and document the previous values.

<div class="page-break"></div>

# 13. Troubleshooting

| Problem | What to try |
|---|---|
| **I cannot sign in** | Check Caps Lock and username spelling. Use password reset if available; generated student accounts should ask the teacher to verify the credential. |
| **A menu is missing** | Your role may not have access. Ask an administrator to check role/group membership. |
| **My scenario is not in Student View** | Check its visibility, organization selection, and student-group assignment. |
| **A question shows no choices** | Return to Authoring Tool, open the question activity, and create or repair its answers. |
| **The learner reaches a dead end** | Check the current activity's direct route or every answer's Next Activity. Confirm that the end was not accidental. |
| **An embedded experiment is blank** | Test the catalogue entry, browser permissions, network access, and external service availability. Try its fallback/open link if shown. |
| **An import fails** | Use the downloaded blank template, preserve sheet/column names, and correct each reported row/column error. For ZIPs, keep the `.xlsx` at the root. |
| **Dashboard charts stay empty** | Select a scenario and Apply. Broaden the date range, remove group filters, and confirm that learners have completed activities. |
| **Metrics/AI results look unreliable** | Check the implementation count. Below the configured threshold, use results only as weak evidence. |
| **AI generation fails or produces malformed content** | Keep the proposal Pending or reject it as structurally invalid, then ask an administrator to inspect the LLM endpoint and structural-failure logs. |
| **A personalised scenario does not appear immediately** | It is created in the background. Wait briefly, refresh Authoring Tool, then ask an administrator to check the worker if it never appears. |
| **My change was not saved** | Look for a validation message, required field, or loading state. Avoid double-clicking Submit. |

When reporting a problem, include the page, scenario name, activity name, approximate time, what you expected, what happened, and a screenshot that does not expose credentials or private learner data.

<div class="page-break"></div>

# 14. Checklists and glossary

## Teacher pre-publication checklist

- [ ] Scenario title, learning goals, age range, subject, language, and duration are accurate.
- [ ] Visibility is still Private during testing.
- [ ] The start activity is an appropriate introduction.
- [ ] Every phase and activity name is clear.
- [ ] Every question has answers and exactly one correct answer where required.
- [ ] Direct, answer-based, and performance routes have been tested.
- [ ] Embedded resources work on learner devices.
- [ ] Evaluation thresholds and branches are sensible.
- [ ] RAG documents are current and authorised.
- [ ] The complete scenario was tested in Student View.
- [ ] The correct student group and visibility were selected.
- [ ] A backup was exported before major deployment.

## Post-session checklist

- [ ] Review dashboard patterns using the correct group and date range.
- [ ] Compare analytics with learner feedback and classroom observation.
- [ ] Investigate activities marked at risk.
- [ ] Review AI proposals one by one; edit before accepting when needed.
- [ ] Export and protect any required records.
- [ ] Improve a private copy before changing a live public scenario.

## Glossary

| Term | Meaning |
|---|---|
| **Activity** | One learner-facing step in a scenario. |
| **Activity route** | A connection from one activity to the next. |
| **Evaluatable activity** | A step that groups performance evidence and routes by thresholds. |
| **Implementation** | A learner attempt/run used in scenario metrics. |
| **LLM context** | Structured scenario information prepared for the language model. |
| **Personalised scenario** | A private copy containing the reviewer's accepted proposals. |
| **Phase** | A major pedagogical stage containing activities. |
| **Proposal** | An AI-generated Create, Revise, or Skip recommendation. |
| **RAG document** | A scenario PDF used as reference context during AI generation. |
| **Risk flag** | A data-derived indicator that an activity may need investigation. |
| **Scenario** | The complete learning experience, its phases, activities, and routes. |
| **Student group** | Generated learner accounts plus assigned scenarios. |

---

**Need more help?** Open **Documentation & Tutorials** inside Trust AI Lab for current support resources and the issue-reporting form. Interface labels can evolve; this guide describes the platform as of July 2026.
