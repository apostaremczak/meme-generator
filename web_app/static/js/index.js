const GeneratorEndpoint = "generate"

const loadingSrc = "../static/images/loader.gif"
const resultImage = document.getElementById("result-image")
const categories = document.querySelectorAll(".category-image")

// console.log(categories)
categories.forEach(category => {
    category.addEventListener("click", () => {
        resultImage.src = loadingSrc

        fetch(`${GeneratorEndpoint}/${category.id}`)
            .then(res => res.json())
            .then(data => resultImage.src = data.result_image)
        window.scrollTo({ top: 0, behavior: 'smooth' })
    })
})
