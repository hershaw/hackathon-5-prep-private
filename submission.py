from io import StringIO
import json
#import ml_metrics as metrics
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from django.shortcuts import render
from django.db.models import Max
from django.views.decorators.csrf import csrf_exempt

from django.http import HttpResponseRedirect

from .models import Team, Submission, SubmissionLimit
from .forms import HackathonSubmissionForm

import sklearn.preprocessing.LabelEncoder
import sklearn.metrics.f1_score as f1_score

y_true_df  = pd.read_csv('./y_true.csv', encoding='utf8')

le = LabelEncoder()
le.fit(y_true_df['newsgroup'].values)

y_true = le.transform(y_true_df['newsgroup'].values)

def index(request):
    leaderboard = [
        {
            'team': Team.objects.get(id=submission['team']),
            'score': submission['score__max'],
        }
        for i, submission in enumerate(
            Submission.objects.values('team').annotate(
                Max('score')).order_by('-score__max'))
        ]

    context = {
        'leaderboard': leaderboard,
        'by_team': [
            (team, Submission.objects.filter(team=team).order_by('number'))
            for team in Team.objects.all()
            ],
    }
    return render(request, 'leaderboard.html', context)


@csrf_exempt
def make_submission(request):

    if request.method == 'GET':
        return render(
            request, 'submission.html', {'form': HackathonSubmissionForm()})

    form = HackathonSubmissionForm(request.POST, request.FILES)
    # form.is_valid()
    if not form.is_valid():
        return render(request, 'submission.html', {'form': form})

    teams = Team.objects.filter(key=form.cleaned_data['team_key'])

    if not teams.count():
        return render(request, 'submission.html', {
            'form': form, 'error': 'Team Key not found'
        })

    team = teams[0]
    limit = SubmissionLimit.objects.get(env='production').limit
    submission_count = Submission.objects.filter(team=team).count()

    if submission_count + 1 > limit:
        return render(request, 'submission.html', {
            'form': form,
            'error': 'Submission count exceeded. You trying to overfit???'
        })
    try:
        csv_obj = StringIO((request.FILES['submission'].read().decode()))
        score = _grade_submission(pd.read_csv(csv_obj))
    except Exception as e:
        print(e)
        return render(request, 'submission.html', {
            'form': form,
            'error': str(e)
        })

    s = Submission(team=team, score=score, number=submission_count + 1)
    s.save()

    return HttpResponseRedirect('/')


def _grade_submission(y_pred):
    y_pred = le.transform(y_pred['newsgroup'].values)

    if y_pred.shape != y_true.shape:
        msg = 'Submission has incorrect shape. Given {}, expected {}'
        raise RuntimeError(msg.format(y_pred.shape, y_true.shape))

    for index, entry in enumerate(y_pred['newsgroup'].values):
        assert_valid(index, entry)

    return f1_score(y_true, y_pred)


def assert_valid(index, entry):

    valid_label_names = set(y_true_df['newsgroup'].unique())

    if type(y_pred) == int:
        raise Exception("Label {index} ({entry}) is an integer instead of a string.".format(index=index, entry=entry))

    if entry not in valid_label_names:
        raise Exception("Label {index} ({entry}) is not a valid label.".format(index=index, entry=entry))
