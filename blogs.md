---
layout: default
title: Blogs
description: "Blog posts by Houquan Zhou covering machine learning, natural language processing, and research insights"
keywords: "blog, machine learning, NLP, natural language processing, research, PhD, Houquan Zhou"
permalink: /blogs/
---

<section class="content-section" style="padding-top: 5rem;">
<h2 class="section-title">Writing</h2>

<div class="blog-list">
  {% assign post_count = 0 %}
  {% for post in site.posts %}
    {% unless post.published == false or post.hidden == true %}
    <article class="blog-entry reveal" data-delay="{{ post_count | times: 60 }}">
      <div class="blog-entry-date">
        {{ post.date | date: "%b %d, %Y" }}
      </div>
      <div class="blog-entry-body">
        <h3 class="blog-entry-title">
          <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
        </h3>
        {% if post.tags %}
          <div class="blog-entry-tags">
            {% for tag in post.tags %}
              <span class="blog-tag">{{ tag }}</span>
            {% endfor %}
          </div>
        {% endif %}
      </div>
    </article>
    {% assign post_count = post_count | plus: 1 %}
    {% endunless %}
  {% endfor %}

  {% if post_count == 0 %}
    <p class="no-posts">No blog posts yet. Stay tuned.</p>
  {% endif %}
</div>
</section>

<style>
  .blog-list {
    margin-top: 0.5rem;
  }

  .blog-entry {
    display: flex;
    gap: 1.5rem;
    align-items: baseline;
    padding: 1.2rem 0;
    border-bottom: 1px solid var(--rule-light);
  }

  .blog-entry:last-child {
    border-bottom: none;
  }

  .blog-entry-date {
    font-size: 0.8rem;
    font-style: italic;
    color: var(--text-light);
    white-space: nowrap;
    min-width: 90px;
  }

  .blog-entry-title {
    font-family: var(--font-display);
    font-size: 1.2rem;
    font-weight: 600;
    line-height: 1.35;
    margin-bottom: 0.3rem;
  }

  .blog-entry-title a {
    color: var(--ink);
    text-decoration: none;
    transition: color 0.25s;
  }

  .blog-entry-title a:hover {
    color: var(--terracotta);
  }

  .blog-entry-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }

  .blog-tag {
    font-size: 0.72rem;
    font-style: italic;
    color: var(--text-light);
  }

  .blog-tag::before {
    content: '#';
  }

  .no-posts {
    text-align: center;
    color: var(--text-light);
    font-style: italic;
    margin: 3rem 0;
  }

  @media screen and (max-width: 480px) {
    .blog-entry {
      flex-direction: column;
      gap: 0.3rem;
    }
    .blog-entry-date {
      min-width: auto;
    }
  }
</style>
