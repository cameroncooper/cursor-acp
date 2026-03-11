// Intercepts child_process.spawn to stream command output to the cursor-acp
// proxy for real-time terminal display. Loaded via NODE_OPTIONS=--require.
'use strict';

(function () {
  var SOCK = process.env.CURSOR_ACP_PTY_SOCK;
  var DEBUG = process.env.CURSOR_ACP_PTY_DEBUG === '1';
  var _logFd;
  function log() {
    if (!DEBUG) return;
    if (!_logFd) {
      try { _logFd = require('fs').openSync(require('path').join(require('os').tmpdir(), 'pty-interceptor.log'), 'a'); } catch(_) { return; }
    }
    var args = Array.prototype.slice.call(arguments);
    require('fs').writeSync(_logFd, '[pty-interceptor pid=' + process.pid + '] ' + args.join(' ') + '\n');
  }
  log('init SOCK=' + SOCK + ' pid=' + process.pid);
  if (!SOCK) { log('missing SOCK env, skipping'); return; }

  var net = require('net');
  var cp = require('child_process');
  var _spawn = cp.spawn;

  var conn = null;
  var connected = false;
  var queue = [];

  function connect() {
    conn = net.createConnection(SOCK);
    conn.on('connect', function () {
      connected = true;
      log('connected to socket');
      for (var i = 0; i < queue.length; i++) conn.write(queue[i]);
      queue = [];
    });
    conn.on('error', function (e) {
      log('socket error: ' + e.message);
      connected = false;
      conn = null;
    });
    conn.on('close', function () {
      log('socket closed');
      connected = false;
      conn = null;
    });
    conn.unref();
  }
  connect();

  function sendFrame(type, pid, data) {
    var pidBuf = Buffer.alloc(4);
    pidBuf.writeInt32BE(pid, 0);
    var payload;
    if (data) {
      var dataBuf = Buffer.isBuffer(data) ? data : Buffer.from(data);
      payload = Buffer.concat([Buffer.from([type]), pidBuf, dataBuf]);
    } else {
      payload = Buffer.concat([Buffer.from([type]), pidBuf]);
    }
    var header = Buffer.alloc(4);
    header.writeUInt32BE(payload.length, 0);
    var frame = Buffer.concat([header, payload]);

    if (connected && conn) {
      conn.write(frame);
    } else {
      queue.push(frame);
      if (!conn) connect();
    }
  }

  cp.spawn = function (command, args, options) {
    var child = _spawn.apply(this, arguments);

    var pid = child.pid;
    if (!pid) return child;

    var cmdStr = '';
    if (Array.isArray(args)) {
      // cursor-agent spawns: /bin/zsh -c "<wrapper>" -- "<actual command>"
      // The user command is after "--", used as $1 in the wrapper script.
      var dashIdx = args.indexOf('--');
      if (dashIdx >= 0 && dashIdx + 1 < args.length) {
        cmdStr = args[dashIdx + 1];
      } else {
        var cIdx = args.indexOf('-c');
        if (cIdx < 0) cIdx = args.indexOf('/c');
        if (cIdx < 0) cIdx = args.indexOf('/C');
        if (cIdx < 0) cIdx = args.indexOf('-Command');
        if (cIdx >= 0 && cIdx + 1 < args.length) {
          cmdStr = args[cIdx + 1];
        } else {
          cmdStr = command + ' ' + args.join(' ');
        }
      }
    }

    log('spawn pid=' + pid + ' cmd=' + command + ' args=' + JSON.stringify(args));
    sendFrame(0x01, pid, cmdStr);

    if (child.stdout) {
      child.stdout.on('data', function (chunk) {
        log('stdout data pid=' + pid + ' len=' + chunk.length + ' connected=' + connected);
        sendFrame(0x02, pid, chunk);
      });
    }

    if (child.stderr) {
      child.stderr.on('data', function (chunk) {
        log('stderr data pid=' + pid + ' len=' + chunk.length);
        sendFrame(0x02, pid, chunk);
      });
    }

    child.on('close', function () {
      log('exit pid=' + pid);
      sendFrame(0x03, pid, null);
    });

    return child;
  };
})();
