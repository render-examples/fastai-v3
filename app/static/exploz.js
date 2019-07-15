

var width = 2600;
var height = 900;

var game = new Phaser.Game(width, height, Phaser.AUTO, 'phaser-example', { preload: preload, create: create, update: update });

var sprite;
var deadGroup = null;

function preload() {
    game.load.crossOrigin = 'Anonymous';
    game.load.spritesheet('ms', 'https://jjwallace.github.io/assets/examples/images/boom.png', 256, 256, 64);
}

function create() {
  createSprite(random(0,width),random(0,height),randomRotation(), Math.floor(Math.random() * 30)   );
}

function createSprite(x, y, r, f) {
    deadGroup = game.add.group();
    sprite = game.add.sprite(40, 100, 'ms');
    sprite.x = x; sprite.y = y;
    sprite.angle = r;
    var anim = sprite.animations.add('boom');
    anim.frame = f;   
    anim.play('boom', 60, false);
    anim.killOnComplete = true;
    anim.onComplete.add(function() {
      deadGroup.add(this);
    }, sprite);
}

function update() {
  if(deadGroup != null){
    deadGroup.forEach(function(sprite) {
      console.debug('destroying:', sprite.name);
      sprite.destroy();
      console.log('destroyed: ', sprite)
    });
  }
  createSprite(random(0,width),random(0,height),randomRotation(), 0);
}

function random(min, max){
  return Math.floor((Math.random() * max) + min);
}

function randomRotation(){
  return Math.floor((Math.random() * 180) - 180);
}