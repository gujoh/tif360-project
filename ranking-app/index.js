const Type = Object.freeze({
    DC: Symbol("dc"),
    CYCLE: Symbol("cycle")
});

class Image {
    constructor(type, id, dcImages, cycleImages) {
        this.type = type;
        this.element = document.getElementById(id);
        this.dcImages = dcImages;
        this.cycleImages = cycleImages;
        this.newImg();
    }

     newImg() {
        var img;
        if (this.type == Type.CYCLE) {
            this.type = Type.DC ;
            var idx = getRandomInt(len);
            img = dcPath + idx + ".png"
        }
        else if (this.type == Type.DC) {
            this.type = Type.CYCLE;
            var idx = getRandomInt(len);
            img = cyclePath + idx + ".png"
        }
        else {
            throw new Error("this.type has an illegal value: " + this.type);
        }
        this.element.src = img;
    }

    increment() {
        if (this.type == Type.CYCLE) {
            cycleCount += 1;
            cycleElement.innerText = "Cycle:" + cycleCount;
        }
        else if (this.type == Type.DC) {
            dcCount += 1;
            dcElement.innerText = "DC:" + dcCount;
        }
        else {
            throw new Error("this.type has an illegal value: " + this.type);
        }
        totalElement.innerText = "Total:" + (dcCount + cycleCount);
    }
}

function getRandomInt(max) {
    return Math.floor(Math.random() * max);
}

len = 2;
cycleCount = 0;
dcCount = 0;
let cycleElement = document.getElementById("cycle_txt")
let dcElement = document.getElementById("dc_txt")
let totalElement = document.getElementById("total_txt")
let dcPath =  "./images/dcgan/dc";
let cyclePath = "./images/cyclegan/cycle"
let leftImage = new Image(Type.DC, "left_img");
let rightImage = new Image(Type.CYCLE, "right_img");

function voteLeft() {
    leftImage.increment();
    leftImage.newImg();
    rightImage.newImg();
}

function voteRight() {
    rightImage.increment();
    leftImage.newImg();
    rightImage.newImg();
}
