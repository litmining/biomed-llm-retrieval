EXAMPLES = [
    {
        "input": (
            "This study used both fMRI-BOLD and structural MRI modalities to investigate the neural correlates "
            "of cognitive control processes. Participants performed two tasks inside the MRI scanner: a Go/No-Go "
            "task (4 blocks, each containing 30 trials, with each trial lasting 2 seconds and a 1-second ITI) "
            "and a short 5-minute resting-state scan (eyes open, instructed to fixate on a crosshair). \n Outside "
            "the scanner, they completed a behavioral Memory Recall task. In the Go/No-Go task, subjects pressed "
            "a button for 'go' stimuli and withheld responses for 'no-go' stimuli. The Memory Recall task involved "
            "seeing a list of 20 words, doing a 30-second distractor task, then recalling as many words as possible "
            "within one minute."
        ),
        "output": '{"Modality": ["fMRI-BOLD", "StructuralMRI"], "StudyObjective": "Investigate the neural correlates of cognitive control processes using fMRI tasks and behavioral assessments.", "Exclude": null, "fMRITasks": [{"TaskName": "Go/No-Go Task", "TaskDescription": "A classic inhibitory control task measuring both reaction time and error rates in response to no-go stimuli.", "DesignDetails": "In this study, the Go/No-Go task included 4 blocks, each with 30 trials. Each trial lasted 2 seconds, with a 1-second inter-trial interval. Subjects were instructed to press a button upon seeing \'go\' stimuli and to refrain from pressing for \'no-go\' stimuli.", "Conditions": ["Go", "No-Go"], "TaskMetrics": ["accuracy", "reaction_time", "fMRI BOLD signal"], "RestingState": false, "TaskDesign": ["Blocked"], "TaskDuration": "10 minutes"}, {"TaskName": "Resting State Scan", "TaskDescription": "Subjects rested with eyes open for baseline functional connectivity measurements.", "DesignDetails": "Participants completed a 5-minute resting-state scan in a single continuous run, instructed to keep their eyes open and minimize movement.", "Conditions": null, "TaskMetrics": ["fMRI BOLD signal"], "RestingState": true, "TaskDesign": ["Other"], "TaskDuration": "5 minutes"}], "BehavioralTasks": [{"TaskName": "Memory Recall Task", "TaskDescription": "A short-term memory task that measures recall of recently presented words.", "DesignDetails": "Each participant saw a list of 20 words (one word every 2 seconds), followed by a 30-second distractor task. Then, they had 1 minute to recall as many words as possible.", "Conditions": ["Word List A", "Word List B"], "TaskMetrics": ["accuracy", "response time"]}]}'
    }
]

EXAMPLESNOMC = [
    {
        "input": (
            "This study used both fMRI-BOLD and structural MRI modalities to investigate the neural correlates "
            "of cognitive control processes. Participants performed two tasks inside the MRI scanner: a Go/No-Go "
            "task (4 blocks, each containing 30 trials, with each trial lasting 2 seconds and a 1-second ITI) "
            "and a short 5-minute resting-state scan (eyes open, instructed to fixate on a crosshair). \n Outside "
            "the scanner, they completed a behavioral Memory Recall task. In the Go/No-Go task, subjects pressed "
            "a button for 'go' stimuli and withheld responses for 'no-go' stimuli. The Memory Recall task involved "
            "seeing a list of 20 words, doing a 30-second distractor task, then recalling as many words as possible "
            "within one minute."
        ),
        "output": '{"StudyObjective": "Investigate the neural correlates of cognitive control processes using fMRI tasks and behavioral assessments.", "fMRITasks": [{"TaskName": "Go/No-Go Task", "TaskDescription": "A classic inhibitory control task measuring both reaction time and error rates in response to no-go stimuli.", "DesignDetails": "In this study, the Go/No-Go task included 4 blocks, each with 30 trials. Each trial lasted 2 seconds, with a 1-second inter-trial interval. Subjects were instructed to press a button upon seeing \'go\' stimuli and to refrain from pressing for \'no-go\' stimuli.", "Conditions": ["Go", "No-Go"], "TaskMetrics": ["accuracy", "reaction_time", "fMRI BOLD signal"], "RestingState": false, , "TaskDuration": "10 minutes"}, {"TaskName": "Resting State Scan", "TaskDescription": "Subjects rested with eyes open for baseline functional connectivity measurements.", "DesignDetails": "Participants completed a 5-minute resting-state scan in a single continuous run, instructed to keep their eyes open and minimize movement.", "Conditions": null, "TaskMetrics": ["fMRI BOLD signal"], "RestingState": true, "TaskDuration": "5 minutes"}], "BehavioralTasks": [{"TaskName": "Memory Recall Task", "TaskDescription": "A short-term memory task that measures recall of recently presented words.", "DesignDetails": "Each participant saw a list of 20 words (one word every 2 seconds), followed by a 30-second distractor task. Then, they had 1 minute to recall as many words as possible.", "Conditions": ["Word List A", "Word List B"], "TaskMetrics": ["accuracy", "response time"]}]}'
    }
]
