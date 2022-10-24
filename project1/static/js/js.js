function() {
    function n(n, e, t) {
        return n.getAttribute(e) || t
    }
    function e(n) {
        return document.getElementsByTagName(n)
    }
    function t() {
        var t = e("script"),
        o = t.length,
        i = t[o - 1];
        return {
            l: o,
            z: n(i, "zIndex", -1),
            o: n(i, "opacity", .5),
            c: n(i, "color", "0,0,0"),
            n: n(i, "count", 99)
        }
    }
    function o() {
        a = m.width = window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth,
        c = m.height = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight
    }
    function i() {
        r.clearRect(0, 0, a, c);
        var n, e, t, o, m, l;
        s.forEach(function(i, x) {
            for (i.x += i.xa, i.y += i.ya, i.xa *= i.x > a || i.x < 0 ? -1 : 1, i.ya *= i.y > c || i.y < 0 ? -1 : 1, r.fillRect(i.x - .5, i.y - .5, 1, 1), e = x + 1; e < u.length; e++) n = u[e],
            null !== n.x && null !== n.y && (o = i.x - n.x, m = i.y - n.y, l = o * o + m * m, l < n.max && (n === y && l >= n.max / 2 && (i.x -= .03 * o, i.y -= .03 * m), t = (n.max - l) / n.max, r.beginPath(), r.lineWidth = t / 2, r.strokeStyle = "rgba(" + d.c + "," + (t + .2) + ")", r.moveTo(i.x, i.y), r.lineTo(n.x, n.y), r.stroke()))
        }),
        x(i)
    }
    var a, c, u, m = document.createElement("canvas"),
    d = t(),
    l = "c_n" + d.l,
    r = m.getContext("2d"),
    x = window.requestAnimationFrame || window.webkitRequestAnimationFrame || window.mozRequestAnimationFrame || window.oRequestAnimationFrame || window.msRequestAnimationFrame ||
    function(n) {
        window.setTimeout(n, 1e3 / 45)
    },
    w = Math.random,
    y = {
        x: null,
        y: null,
        max: 2e4
    };
    m.id = l,
    m.style.cssText = "position:fixed;top:0;left:0;z-index:" + d.z + ";opacity:" + d.o,
    e("body")[0].appendChild(m),
    o(),
    window.onresize = o,
    window.onmousemove = function(n) {
        n = n || window.event,
        y.x = n.clientX,
        y.y = n.clientY
    },
    window.onmouseout = function() {
        y.x = null,
        y.y = null
    };
    for (var s = [], f = 0; d.n > f; f++) {
        var h = w() * a,
        g = w() * c,
        v = 2 * w() - 1,
        p = 2 * w() - 1;
        s.push({
            x: h,
            y: g,
            xa: v,
            ya: p,
            max: 6e3
        })
    }
    u = s.concat([y]),
    setTimeout(function() {
        i()
    },
    100)
} ();
    var box = document.getElementById("box");
    var oNavlist = document.getElementById("nav").children;
    var slider = document.getElementById("slider");
    var left = document.getElementById("left");
    var right = document.getElementById("right");
    var tpp = document.getElementById("tpp");
    var index = 1;
    var timer;
    var isMoving = false;
    //轮播下一张的函数
    function next(){
        if(isMoving){
            return;
        }
        isMoving = true;
        index++;
        navChange();
        animate(slider,{left:-1200*index},function(){
            if(index > 3){
                slider.style.left="-1200px";
                index = 1;
            }
            isMoving = false;
        });
    }
    function prev(){
        if(isMoving){
            return;
        }
        index--;
        navChange();
        animate(slider,{left:-1200*index},function(){
            if(index === 0){
                slider.style.left="-6000px";
                index = 3;
            }
            isMoving = false;
        });
    }
    timer = setInterval(next,2000);
    //鼠标划入清定时器
    box.onmouseover = function(){
        animate(left,{opacity:50});
        animate(right,{opacity:50});
        clearInterval(timer);
    }
    //鼠标划出开定时器
    box.onmouseout = function(){
        animate(left,{opacity:0});
        animate(right,{opacity:0});
        timer = setInterval(next, 2000);
    }
    right.onclick = next;
    left.onclick = prev;
    //小按钮点击事件
    for(var i=0;i<oNavlist.length;++i){
        oNavlist[i].idx = i;
        oNavlist[i].onclick = function(){
            index = this.idx + 1;
            navChange();
            animate(slider,{left:-1200*index});
        }
    }
    //小按钮背景颜色
    function navChange(){
        for(var i = 0;i<oNavlist.length;++i){
            oNavlist[i].className = " ";
        }
        if(index>3){
            oNavlist[0].className = "active";
        }else if(index === 0){
            oNavlist[2].className = "active";
        }else{
            oNavlist[index-1].className = "active";
        }
    }
    var j = 800;
    setInterval(function(){
        tpp.style.left = j + "px";
        j--;
        if(j<-350){
            j=800;
        }
    }, 10);