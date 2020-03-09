from flask import (Blueprint, redirect, render_template, request, session, url_for)
from .io import write_data, write_metadata
from .utils import gen_code, compute_bonus

## Initialize blueprint.
bp = Blueprint('complete', __name__)

@bp.route('/complete')
def complete():
    """Present completion screen to participant."""

    ## DEV NOTE:
    ## If you want a custom completion code, replace the return statement with:
    ## > render_template('complete.html', value=session['complete'])

    return render_template('complete.html')

@bp.route('/datadump', methods = ['POST'])
def datadump():
    """Save jsPsych data to disk."""

    if request.is_json:

        ## Retrieve jsPsych data.
        JSON = request.get_json()

        ## Save jsPsch data to disk.
        write_data(session, JSON)

        ## Update participant metadata.
        session['complete'] = True
        write_metadata(session, ['complete'], 'a')

        return redirect(url_for('complete.complete'))

    else:

        ## Update participant metadata.
        session['complete'] = True
        session['ERROR'] = "1011: Noncompliant behavior."
        write_metadata(session, ['complete','ERROR'], 'a')

        ## Redirect participant to error (previous participation).
        return redirect(url_for('error.error', errornum=1011))
