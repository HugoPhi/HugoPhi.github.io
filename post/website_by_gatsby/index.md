# My Website & blog

中文：[zh](./index_zh.html)

[TOC]



**Link**: [Yunming Hu's Personal Page](https://ohpostintm3.top)

## How it works?

​	My website is a static site generated using [gatsby-themes](https://github.com/LekoArts/gatsby-themes) and deployed on GitHub Pages, while my blog is handwritten and also deployed on GitHub Pages. If you're unfamiliar with how to use GitHub Pages, I recommend checking out this [quickstart article](https://docs.github.com/en/pages/quickstart) for guidance. 



## Procedure of deploy

### Website

#### install 

- clone the project and rm its .git and add your own: 

```bash
git clone https://github.com/LekoArts/gatsby-starter-portfolio-cara.git
cd gatsby-starter-portfolio-cara
rm -rf .git
git init 
git remote add origin 'your github page'
```

- install dependnecies:

```bash
npm install
npm install gh-pages --save-dev
```

- add script to `pacakge.json`: 

```json
"scripts": {
    "develop": "gatsby develop",
    "deploy": "gatsby build --prefix-paths && gh-pages -d public",
    "build": "gatsby build --prefix-paths"
}
```

and modify `gatsby-config.ts`, here is a template you can change: 

```json
siteMetadata: {
  siteTitle: `Your Site Title`,
  siteTitleAlt: `Your Site Subtitle`,
  siteHeadline: `A brief description about your site`,
  siteUrl: `https://yourdomain.com`,
  siteDescription: `A description of your site, suitable for SEO`,
  siteImage: `/your-image.jpg`, // Default image for social sharing
  siteLanguage: `en`, // Language setting
  author: `Your Name or Nickname`,
}
```

- build & deploy
- set branch `gh-pages` as branch to deploy, just ask GPT for help. 
- then add your domain into file CNAME ubder public, create it if not exsited.
- rebuild and deploy
- the you can see the template on your domain.

#### Project structure & Commands

​	The website we deploy is generated using gatsby-themes, and the process is simple: just edit the .mdx files and run a few commands. The entire website is divided into four sections: Introduction, Projects, About, and Get in Touch. To edit the content in these sections, you can simply modify the .mdx files located in the folder `$project/src/@lekoarts/gatsby-theme-cara/sections`, where `$project` refers to your project folder. When you edit and build the project, the four files will be converted into .html and added to index.html. You can then deploy it using the command: `npm run deploy`. Here are all the commands and their usage:

##### npm run build

​	This command builds the entire project and places the generated files into the `$project/public`. It's important to note a key point that is easy to overlook: you need to add a CNAME file to the public folder that contains your own domain name. For example, my domain is `ohpostint.top`, so you should include this domain in the CNAME file, as shown in the figure below:

![CNAME in public folder](./asset/1.png)

![content in CNAME](./asset/2.png)

​	And before you build it, you can choose whether to clean the previous built files or not, whose command is `gatsby clean`. 



##### npm run develop 

​	After building your project, you can preview it locally before deploying it online. This command lets you preview the site while editing your `.mdx` section files in real-time. For example, after running this command, you can click the [local server link](http://localhost:8000/) to view the website in your browser. When you make changes to an `.mdx` file, take `intro.mdx` for instance, the updates will appear instantly in your browser, as shown in the video below: 

![develop](./asset/develop.gif)

##### npm run deploy

​	After debugging, you can deploy your website to your domain use this command.

### Blog

​	My blog depends on no code generator. What you should to is just write markdown files in typora and export them as index.html & index_zh.html under the path. For example, you want to write an article about PCA algorithm, this is steps you should do:

##### 1. create a folder to contain this article as a project

​	In my blog, an article is organized as a project that means you should put anything needed in your blog into its folder, and all of these folders are located in `$project/post/`. In this example, I name it `pca`: 

```bash
mkdir ./post/pca
cd ./post/pca
```

##### 2. create index.md & edit it

​	After you get into project of this article, you can start writing your blog. If you want to use picture, you should put them into `$project/post/pca/asset/`. 

##### 3. convert to index.html

​	After you ending editing, you should convert your index.md to index.html. I use the export function in **Typora**.

##### 4. refer it in top index.md

​	Then you can edit index.md under project root, refering this article in it. And convert index.md under project root to index.html.

##### 5. Multilingual support

​	You can create `index_xx.md` files for multilingual support, where `xx` represents your target language. For example, for Chinese: `index_zh.md`. Then, repeat steps 2 to 4. Additionally, you need to insert links to other language versions at the top of each `index` file. For instance, in the English `index.md`, you should insert:

```markdown
中文版：[zh](./index_zh.md)
```

and Iin the Chinese `index_zh.md`, you should insert:

```markdown
English: [en](./index.md)
```

Due to the structure of our organization, these lines are almost fixed. This is one of the reasons why I chose this structure—it’s somewhat similar to a B+ tree.

## Insight of Blog design

### How I structure the project?

​	Home page is editing in index.html under project root. And articles are located in `$project/post/`.  All article is named as 'index_*.html', '\*' here is used to supprt multilingual versions. So you can refer all article easily. In each article folder, you can create anything of this article such as pictures, anmations and etc. without influnece the others.

### Where does the design of this style come from?

​	This style is inspired by this [website](https://sites.math.washington.edu//~morrow/mcm/mcm.html) & typora theme [Turing](https://theme.typora.io/theme/Turing/), which is in line with minimalist style. I will provide it in [index.css](./css/index.css) & [hugo](./css/hugo.css). 