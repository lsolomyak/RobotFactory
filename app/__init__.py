import os, sys, configparser, warnings
from flask import (Flask, redirect, render_template, request, session, url_for)
from app import consent, alert, experiment, complete, error
from .io import write_metadata
from .utils import gen_code

## Define root directory.
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

## Load and parse configuration file.
cfg = configparser.ConfigParser()
cfg.read(os.path.join(ROOT_DIR, 'app.ini'))

## Ensure output directories exist.
data_dir = os.path.join(ROOT_DIR, cfg['IO']['DATA'])
if not os.path.isdir(data_dir): os.makedirs(data_dir)
meta_dir = os.path.join(ROOT_DIR, cfg['IO']['METADATA'])
if not os.path.isdir(meta_dir): os.makedirs(meta_dir)

## Check Flask password.
if cfg['FLASK']['SECRET_KEY'] == "PLEASE_CHANGE_THIS":
    msg = "WARNING: Flask password is currently default. This should be changed prior to production."
    warnings.warn(msg)

## Initialize Flask application.
app = Flask(__name__)
app.secret_key = cfg['FLASK']['SECRET_KEY']

## Apply blueprints to the application.
app.register_blueprint(consent.bp)
app.register_blueprint(alert.bp)
app.register_blueprint(experiment.bp)
app.register_blueprint(complete.bp)
app.register_blueprint(error.bp)

## Define root node.
@app.route('/')
def index():

    ## Store directories in session object.
    session['data'] = data_dir
    session['metadata'] = meta_dir

    ## Record incoming metadata.
    info = dict(
        workerId     = request.args.get('workerId'),        # MTurk metadata
        assignmentId = request.args.get('assignmentId'),    # MTurk metadata
        hitId        = request.args.get('hitId'),           # MTurk metadata
        subId        = gen_code(24),                        # NivTurk metadata
        a            = request.args.get('a'),               # TurkPrime metadata
        tp_a         = request.args.get('tp_a'),            # TurkPrime metadata
        b            = request.args.get('b'),               # TurkPrime metadata
        tp_b         = request.args.get('tp_b'),            # TurkPrime metadata
        c            = request.args.get('c'),               # TurkPrime metadata
        tp_c         = request.args.get('tp_c')             # TurkPrime metadata
    )

    ## Error-catching: screen for previous session.
    if 'workerId' in session:

        ## Define error message.
        if info['workerId'] is not None and session['workerId'] != info['workerId']:
            errmsg = "1004: Revisited index. [WARNING] workerId tampering detected."
        else:
            errmsg = "1004: Revisited index."

        ## Update metadata.
        session['ERROR'] = errmsg
        write_metadata(session, ['ERROR'], 'a')

        ## Redirect participant to error (previous participation).
        return redirect(url_for('error.error', errornum=1004))

    ## Error-catching: screen for valid workerId.
    elif info['workerId'] is None:

        ## Redirect participant to error (admin error).
        return redirect(url_for('error.error', errornum=1000))

    ## Error-catching: screen for workerId in database.
    elif info['workerId'] in os.listdir(meta_dir):

        ## Update metadata.
        session['ERROR'] = "1004: Revisited index."
        write_metadata(session, ['ERROR'], 'a')

        ## Redirect participant to error (previous participation).
        return redirect(url_for('error.error', errornum=1004))

    else:

        ## Update metadata.
        for k, v in info.items(): session[k] = v
        write_metadata(session, ['workerId','hitId','assignmentId','subId'], 'w')

        ## Redirect participant to consent form.
        return redirect(url_for('consent.consent'))

## DEV NOTE:
## The following route is strictly for development purposes and should be
## commented out before deployment.
# @app.route('/clear')
# def clear():
#     session.clear()
#     return 'Complete!'
