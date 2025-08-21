---
layout: default
title: Blog
permalink: /blogs/
---

## Blog Posts

<div class="blog-list">
  {% for post in site.posts %}
    {% unless post.published == false or post.hidden == true %}
    <article class="blog-item">
      <h3 class="blog-title">
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </h3>
      <div class="blog-meta">
        <time datetime="{{ post.date | date_to_xmlschema }}">
          {{ post.date | date: "%B %d, %Y" }}
        </time>
        {% if post.author %}
          • {{ post.author }}
        {% endif %}
      </div>
      {% if post.excerpt %}
        <div class="blog-excerpt">
          {{ post.excerpt | strip_html | truncatewords: 50 }}
        </div>
      {% endif %}
      {% if post.tags %}
        <div class="blog-tags">
          {% for tag in post.tags %}
            <span class="blog-tag">{{ tag }}</span>
          {% endfor %}
        </div>
      {% endif %}
    </article>
    {% endunless %}
  {% endfor %}

  {% assign visible_posts = site.posts | where_exp: "post", "post.published != false and post.hidden != true" %}
  {% if visible_posts.size == 0 %}
    <p class="no-posts">No blog posts yet. Stay tuned!</p>
  {% endif %}
</div>

<style>
  .blog-list {
    margin-top: 2rem;
  }

  .blog-item {
    margin-bottom: 2.5rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #e1e1e1;
  }

  .blog-item:last-child {
    border-bottom: none;
  }

  .blog-title {
    margin-bottom: 0.5rem;
    font-size: 1.4rem;
  }

  .blog-title a {
    color: #333;
    text-decoration: none;
    transition: color 0.3s;
  }

  .blog-title a:hover {
    color: #68b88e;
  }

  .blog-meta {
    color: #666;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
  }

  .blog-excerpt {
    color: #555;
    line-height: 1.6;
    margin: 1rem 0;
  }

  .blog-tags {
    margin-top: 0.5rem;
  }

  .blog-tag {
    display: inline-block;
    background-color: #68b88e;
    color: white;
    padding: 0.2rem 0.5rem;
    margin-right: 0.3rem;
    border-radius: 3px;
    font-size: 0.8rem;
  }

  .no-posts {
    text-align: center;
    color: #666;
    font-style: italic;
    margin: 3rem 0;
  }

  @media screen and (max-width: 720px) {
    .blog-title {
      font-size: 1.2rem;
    }
  }
</style> 